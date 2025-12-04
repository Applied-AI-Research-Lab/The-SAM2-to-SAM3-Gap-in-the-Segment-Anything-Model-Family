#!/usr/bin/env python3
"""
SAM2 Model Wrapper

This module provides a clean interface to the SAM2 model for prompt-based
segmentation tasks. SAM2 operates through visual prompts such as points,
bounding boxes, and mask priors, focusing on geometric and temporal consistency.

The wrapper handles model initialization, prompt processing, inference execution,
and result formatting to simplify integration with the experimental pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except ImportError:
    logging.warning("SAM2 not installed. Please install: pip install segment-anything-2")
    SAM2ImagePredictor = None
    SAM2VideoPredictor = None


logger = logging.getLogger(__name__)


class SAM2Wrapper:
    """
    Wrapper class for SAM2 model providing a simplified interface.
    
    This wrapper encapsulates SAM2 functionality for both image and video
    segmentation tasks, handling model initialization, prompt encoding,
    inference execution, and post-processing of segmentation masks.
    
    SAM2 requires explicit visual prompts and operates in a purely geometric
    space without semantic understanding of object categories or attributes.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_file: str,
        device: str = "cuda",
        compile_model: bool = False,
    ):
        """
        Initialize SAM2 model with specified configuration.
        
        Args:
            checkpoint_path: Path to SAM2 model checkpoint file
            config_file: Path to SAM2 configuration YAML file
            device: Device to run model on (cuda, cpu, or mps)
            compile_model: Whether to compile model with torch.compile for speed
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_file = config_file
        self.device = device
        self.compile_model = compile_model
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Initializing SAM2 model from {checkpoint_path}")
        self._load_model()
        
    def _load_model(self):
        """Load and initialize the SAM2 model architecture."""
        if SAM2ImagePredictor is None:
            raise ImportError("SAM2 not available. Please install segment-anything-2")
        
        # Build SAM2 model from configuration
        sam2_model = build_sam2(
            config_file=self.config_file,
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
        )
        
        # Wrap in predictor interface
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        # Optional compilation for faster inference
        if self.compile_model:
            logger.info("Compiling model with torch.compile...")
            self.predictor.model = torch.compile(
                self.predictor.model,
                mode="default"
            )
        
        logger.info("SAM2 model loaded successfully")
    
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Process and encode an image for subsequent prompt-based segmentation.
        
        This method computes image embeddings that will be reused for all
        prompts on this image, enabling efficient multi-prompt inference.
        
        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            Dictionary containing image metadata and encoded features
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.ndim == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        
        logger.debug(f"Processing image of shape {image.shape}")
        
        # Encode image through SAM2 vision backbone
        self.predictor.set_image(image)
        
        return {
            "shape": image.shape,
            "dtype": image.dtype,
            "encoded": True,
        }
    
    def predict_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate segmentation masks from point prompts.
        
        Point prompts are the most basic form of visual guidance in SAM2,
        where each point indicates either foreground (label 1) or background
        (label 0) regions to guide mask generation.
        
        Args:
            point_coords: Array of shape (N, 2) with (x, y) coordinates
            point_labels: Array of shape (N,) with 1 for foreground, 0 for background
            multimask_output: If True, return 3 candidate masks with scores
            
        Returns:
            Dictionary containing:
                - masks: Array of shape (N_masks, H, W) with binary masks
                - scores: Array of shape (N_masks,) with quality scores
                - logits: Low-resolution mask logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
        }
    
    def predict_with_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate segmentation mask from bounding box prompt.
        
        Box prompts provide stronger spatial constraints than points,
        indicating the approximate rectangular region containing the target object.
        
        Args:
            box: Array of shape (4,) with format [x_min, y_min, x_max, y_max]
            multimask_output: If True, return multiple mask candidates
            
        Returns:
            Dictionary with masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output,
        )
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
        }
    
    def predict_with_mask(
        self,
        mask_input: np.ndarray,
        multimask_output: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Refine an existing mask using SAM2 decoder.
        
        Mask prompts allow iterative refinement of segmentation results,
        useful for correcting errors or adapting masks across frames.
        
        Args:
            mask_input: Low-resolution mask logits from previous prediction
            multimask_output: Whether to output multiple refinement candidates
            
        Returns:
            Dictionary with refined masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
        }
    
    def predict_auto_grid(
        self,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Automatically generate masks using a grid of point prompts.
        
        This method implements SAM2's automatic mask generation by sampling
        a regular grid of points across the image and generating masks for
        each point, then filtering and combining results.
        
        Args:
            points_per_side: Number of sample points per image side
            pred_iou_thresh: Minimum predicted IoU for mask filtering
            stability_score_thresh: Minimum stability score for mask quality
            
        Returns:
            List of dictionaries, each containing a mask with its metadata
        """
        # Generate grid of points
        h, w = self.predictor._features["image_embed"].shape[-2:]
        grid_size = points_per_side
        
        x_coords = np.linspace(0, w - 1, grid_size)
        y_coords = np.linspace(0, h - 1, grid_size)
        
        all_masks = []
        
        for x in x_coords:
            for y in y_coords:
                point_coords = np.array([[x, y]])
                point_labels = np.array([1])
                
                result = self.predict_with_points(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                
                # Filter by quality thresholds
                for mask, score in zip(result["masks"], result["scores"]):
                    if score >= pred_iou_thresh:
                        all_masks.append({
                            "mask": mask,
                            "score": score,
                            "point": (x, y),
                        })
        
        logger.info(f"Generated {len(all_masks)} masks from grid sampling")
        return all_masks
    
    def reset(self):
        """Reset the predictor state and clear cached image features."""
        self.predictor.reset_predictor()
        logger.debug("Predictor state reset")
    
    def get_image_embedding(self) -> Optional[torch.Tensor]:
        """
        Retrieve the computed image embedding from the vision backbone.
        
        Returns:
            Image embedding tensor or None if no image has been set
        """
        if hasattr(self.predictor, "_features") and "image_embed" in self.predictor._features:
            return self.predictor._features["image_embed"]
        return None
    
    def to(self, device: str):
        """
        Move model to specified device.
        
        Args:
            device: Target device (cuda, cpu, mps)
        """
        self.device = device
        self.predictor.model.to(device)
        logger.info(f"Model moved to {device}")


class SAM2VideoWrapper:
    """
    Wrapper for SAM2 video segmentation with temporal memory.
    
    This class extends SAM2 functionality to video sequences, leveraging
    temporal memory to propagate masks across frames and maintain object
    identity through occlusions and appearance changes.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_file: str,
        device: str = "cuda",
    ):
        """
        Initialize SAM2 video predictor.
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config_file: Path to configuration file
            device: Device for inference
        """
        if SAM2VideoPredictor is None:
            raise ImportError("SAM2 video predictor not available")
        
        # Build video predictor
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device,
        )
        
        self.predictor = SAM2VideoPredictor(sam2_model)
        self.device = device
        logger.info("SAM2 video predictor initialized")
    
    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
    ) -> Dict:
        """
        Initialize video state and load frames.
        
        Args:
            video_path: Path to video file or directory of frames
            offload_video_to_cpu: Whether to store frames in CPU memory
            
        Returns:
            Inference state dictionary for this video
        """
        inference_state = self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        
        logger.info(f"Initialized video state with {inference_state['num_frames']} frames")
        return inference_state
    
    def add_prompt_points(
        self,
        inference_state: Dict,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Add point prompts for an object on a specific frame.
        
        Args:
            inference_state: Video inference state
            frame_idx: Frame index to add prompts
            obj_id: Object ID for tracking
            points: Point coordinates array (N, 2)
            labels: Point labels (1 for foreground, 0 for background)
        """
        self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
    
    def propagate_in_video(
        self,
        inference_state: Dict,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate masks across all frames using temporal memory.
        
        Args:
            inference_state: Video inference state with initial prompts
            
        Returns:
            Dictionary mapping frame indices to object masks
        """
        video_segments = {}
        
        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                obj_id: mask for obj_id, mask in zip(obj_ids, masks)
            }
        
        logger.info(f"Propagated masks across {len(video_segments)} frames")
        return video_segments
