"""
SAM2 Fine-tuning Script

This script provides complete fine-tuning infrastructure for SAM2 models on custom datasets.
Supports prompt-based segmentation with geometric inputs (points, boxes, masks).

Key Features:
- Full fine-tuning or selective layer training
- Multiple loss functions: IoU loss, focal loss, dice loss
- Checkpoint saving and resumption
- Learning rate scheduling with warmup
- Validation monitoring
- Mixed precision training support

Usage:
    python src/training/finetune_sam2.py --config configs/train_sam2.yml
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 package not installed. Install with: pip install segment-anything-2")
    SAM2_AVAILABLE = False

from utils.dataset_loader import MineAppleDataset, create_data_loaders
from prompts.sam2_prompts import generate_point_prompt, generate_box_prompt


class SAM2FineTuner:
    """
    Fine-tuning manager for SAM2 models
    
    Handles training loop, optimization, checkpointing, and validation
    for prompt-based segmentation tasks
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fine-tuning manager
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.mixed_precision = config['training'].get('mixed_precision', True)
        
        # Model parameters
        self.model_type = config['model']['type']
        self.checkpoint_path = config['model']['checkpoint']
        self.freeze_image_encoder = config['model'].get('freeze_image_encoder', False)
        self.freeze_prompt_encoder = config['model'].get('freeze_prompt_encoder', False)
        
        # Loss parameters
        self.loss_weights = config['loss']
        
        # Output directories
        self.output_dir = Path(config['output']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        
        print(f"Initialized SAM2FineTuner on {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def setup_model(self):
        """
        Load SAM2 model and configure trainable parameters
        """
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not available. Install segment-anything-2 package.")
        
        print(f"\nLoading SAM2 model: {self.model_type}")
        print(f"Checkpoint: {self.checkpoint_path}")
        
        # Load model
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model = self.model.to(self.device)
        
        # Configure trainable parameters
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Freeze image encoder if specified
            if self.freeze_image_encoder and 'image_encoder' in name:
                param.requires_grad = False
            # Freeze prompt encoder if specified
            elif self.freeze_prompt_encoder and 'prompt_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params += param.numel()
        
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
        
        # Set frozen modules to eval mode
        if self.freeze_image_encoder:
            self.model.image_encoder.eval()
            print("  Image encoder: FROZEN (eval mode)")
        if self.freeze_prompt_encoder:
            self.model.prompt_encoder.eval()
            print("  Prompt encoder: FROZEN (eval mode)")
    
    def setup_optimizer(self):
        """
        Configure optimizer and learning rate scheduler
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        warmup_epochs = self.config['training'].get('warmup_epochs', 2)
        
        def lr_lambda(epoch):
            """Learning rate schedule with linear warmup and cosine decay"""
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (self.num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"\nOptimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        print(f"Scheduler: Cosine annealing with {warmup_epochs} warmup epochs")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    iou_predictions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for SAM2 training
        
        Args:
            predictions: Predicted masks [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
            iou_predictions: Predicted IoU scores [B, 1]
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Focal loss for handling class imbalance
        if self.loss_weights.get('focal', 0) > 0:
            focal_loss = self._focal_loss(predictions, targets)
            losses['focal'] = focal_loss * self.loss_weights['focal']
        
        # Dice loss for overlap optimization
        if self.loss_weights.get('dice', 0) > 0:
            dice_loss = self._dice_loss(predictions, targets)
            losses['dice'] = dice_loss * self.loss_weights['dice']
        
        # IoU loss if IoU predictions available
        if iou_predictions is not None and self.loss_weights.get('iou', 0) > 0:
            iou_loss = self._iou_loss(predictions, targets, iou_predictions)
            losses['iou'] = iou_loss * self.loss_weights['iou']
        
        # Binary cross-entropy loss
        if self.loss_weights.get('bce', 0) > 0:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets)
            losses['bce'] = bce_loss * self.loss_weights['bce']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for addressing class imbalance
        
        Focuses training on hard examples by down-weighting easy examples
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** gamma
        
        # Compute binary cross-entropy
        bce = nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Apply focal weight and alpha balancing
        focal_loss = focal_weight * bce
        focal_loss = torch.where(targets == 1, alpha * focal_loss, (1 - alpha) * focal_loss)
        
        return focal_loss.mean()
    
    def _dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for optimizing mask overlap
        
        Directly optimizes the Dice coefficient (F1 score for segmentation)
        """
        # Apply sigmoid
        probs = torch.sigmoid(predictions)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        
        # Return Dice loss
        return 1.0 - dice
    
    def _iou_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                  iou_predictions: torch.Tensor) -> torch.Tensor:
        """
        IoU prediction loss
        
        Trains the model to accurately predict the IoU of its own masks
        """
        # Apply sigmoid to predictions
        probs = torch.sigmoid(predictions)
        
        # Compute actual IoU
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        actual_iou = intersection / (union + 1e-6)
        
        # MSE loss between predicted and actual IoU
        iou_loss = nn.functional.mse_loss(iou_predictions.squeeze(), actual_iou.squeeze())
        
        return iou_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        
        # Set frozen modules to eval mode
        if self.freeze_image_encoder:
            self.model.image_encoder.eval()
        if self.freeze_prompt_encoder:
            self.model.prompt_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            images = batch['image_array'].to(self.device)
            masks = batch['masks'].to(self.device)  # Ground truth masks
            
            # Generate prompts (point or box)
            prompt_type = self.config['training'].get('prompt_type', 'point')
            
            if prompt_type == 'point':
                # Generate point prompts from masks
                points, labels = self._generate_point_prompts_from_masks(masks)
            elif prompt_type == 'box':
                # Generate box prompts from masks
                boxes = self._generate_box_prompts_from_masks(masks)
                points, labels = None, None
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                # Encode image
                with torch.set_grad_enabled(not self.freeze_image_encoder):
                    image_embeddings = self.model.image_encoder(images)
                
                # Encode prompts
                with torch.set_grad_enabled(not self.freeze_prompt_encoder):
                    if prompt_type == 'point':
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=(points, labels),
                            boxes=None,
                            masks=None
                        )
                    else:
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None
                        )
                
                # Decode masks
                mask_predictions, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                
                # Compute loss
                losses = self.compute_loss(mask_predictions, masks, iou_predictions)
                loss = losses['total']
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                images = batch['image_array'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Generate prompts
                prompt_type = self.config['training'].get('prompt_type', 'point')
                
                if prompt_type == 'point':
                    points, labels = self._generate_point_prompts_from_masks(masks)
                else:
                    boxes = self._generate_box_prompts_from_masks(masks)
                    points, labels = None, None
                
                # Forward pass
                image_embeddings = self.model.image_encoder(images)
                
                if prompt_type == 'point':
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=(points, labels),
                        boxes=None,
                        masks=None
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None
                    )
                
                mask_predictions, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                
                # Compute loss
                losses = self.compute_loss(mask_predictions, masks, iou_predictions)
                loss = losses['total']
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _generate_point_prompts_from_masks(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate point prompts from ground truth masks
        
        Samples positive points from mask regions
        """
        batch_size = masks.shape[0]
        points = []
        labels = []
        
        for i in range(batch_size):
            mask = masks[i, 0].cpu().numpy()
            
            # Find positive pixels
            pos_coords = np.argwhere(mask > 0.5)
            
            if len(pos_coords) > 0:
                # Sample random positive point
                idx = np.random.randint(len(pos_coords))
                point = pos_coords[idx]
                points.append([point[1], point[0]])  # x, y format
                labels.append(1)
            else:
                # No positive pixels, use center point
                h, w = mask.shape
                points.append([w // 2, h // 2])
                labels.append(0)
        
        points = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(1)
        labels = torch.tensor(labels, dtype=torch.int32, device=self.device).unsqueeze(1)
        
        return points, labels
    
    def _generate_box_prompts_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Generate box prompts from ground truth masks
        
        Computes bounding boxes from masks
        """
        batch_size = masks.shape[0]
        boxes = []
        
        for i in range(batch_size):
            mask = masks[i, 0].cpu().numpy()
            
            # Find bounding box
            pos_coords = np.argwhere(mask > 0.5)
            
            if len(pos_coords) > 0:
                y_min, x_min = pos_coords.min(axis=0)
                y_max, x_max = pos_coords.max(axis=0)
                boxes.append([x_min, y_min, x_max, y_max])
            else:
                # No positive pixels, use image center
                h, w = mask.shape
                boxes.append([w // 4, h // 4, 3 * w // 4, 3 * h // 4])
        
        boxes = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        return boxes
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest model
        latest_path = self.output_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nStarting training for {self.num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 80)
        
        for epoch in range(self.current_epoch + 1, self.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            if is_best:
                print(f"  ✓ New best model!")
            print("=" * 80)
            
            # Save checkpoint
            if epoch % self.config['output'].get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")


def main():
    """
    Main training entry point
    """
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 on custom dataset")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("SAM2 Fine-tuning")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create fine-tuner
    fine_tuner = SAM2FineTuner(config)
    
    # Setup model and optimizer
    fine_tuner.setup_model()
    fine_tuner.setup_optimizer()
    
    # Resume from checkpoint if specified
    if args.resume:
        fine_tuner.load_checkpoint(args.resume)
    
    # Load data
    data_root = config['data']['root_dir']
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Train
    fine_tuner.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
