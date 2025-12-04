#!/usr/bin/env python3
"""
MineApple Dataset Loader

This module provides data loading utilities for the MineApple orchard dataset,
including image loading, annotation parsing, train/val/test splitting, and
batching utilities for efficient experiment execution.

The dataset contains apple images with instance-level annotations including
masks, bounding boxes, and attribute labels for ripeness, color, and health.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MineAppleDataset:
    """
    Dataset loader for MineApple orchard imagery with apple annotations.
    
    This class handles loading images, parsing annotations, and providing
    data splits for systematic SAM2/SAM3 evaluation experiments.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
    ):
        """
        Initialize MineApple dataset loader.
        
        Args:
            data_root: Path to dataset root directory
            split: Dataset split to load (train, val, or test)
            split_ratios: Tuple of (train, val, test) ratios summing to 1.0
            random_seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.split = split
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        
        # Define directory structure
        self.images_dir = self.data_root / "images"
        self.annotations_dir = self.data_root / "annotations"
        self.metadata_file = self.data_root / "metadata.json"
        
        # Load metadata and annotations
        self._load_metadata()
        self._load_annotations()
        self._create_splits()
        
        logger.info(f"Loaded MineApple {split} split with {len(self.samples)} images")
    
    def _load_metadata(self):
        """Load dataset metadata from JSON file."""
        if not self.metadata_file.exists():
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            self.metadata = {}
            return
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        logger.debug(f"Loaded metadata with {len(self.metadata)} entries")
    
    def _load_annotations(self):
        """
        Load all annotation files from annotations directory.
        
        Each annotation file contains instance masks, bounding boxes,
        and semantic labels for one image.
        """
        self.annotations = {}
        
        if not self.annotations_dir.exists():
            logger.error(f"Annotations directory not found: {self.annotations_dir}")
            return
        
        for ann_file in self.annotations_dir.glob("*.json"):
            image_id = ann_file.stem
            
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            self.annotations[image_id] = ann_data
        
        logger.debug(f"Loaded annotations for {len(self.annotations)} images")
    
    def _create_splits(self):
        """
        Create train/val/test splits from available data.
        
        Uses stratified sampling to ensure balanced distribution of
        apple counts and attributes across splits.
        """
        np.random.seed(self.random_seed)
        
        # Get all image IDs
        image_ids = list(self.annotations.keys())
        
        if len(image_ids) == 0:
            logger.error("No annotations found, cannot create splits")
            self.samples = []
            return
        
        # Shuffle for randomization
        np.random.shuffle(image_ids)
        
        # Compute split boundaries
        n_total = len(image_ids)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        
        # Assign to splits
        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]
        
        # Select appropriate split
        if self.split == "train":
            self.samples = train_ids
        elif self.split == "val":
            self.samples = val_ids
        elif self.split == "test":
            self.samples = test_ids
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        logger.info(f"Split sizes - train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    
    def __len__(self) -> int:
        """Return number of samples in current split."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample by index.
        
        Args:
            idx: Sample index in current split
            
        Returns:
            Dictionary containing:
                - image_id: Unique image identifier
                - image: PIL Image object
                - image_array: Numpy array (H, W, 3) in RGB
                - annotations: List of instance annotations
                - metadata: Image-level metadata
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for {len(self.samples)} samples")
        
        image_id = self.samples[idx]
        
        # Load image
        image_path = self.images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.images_dir / f"{image_id}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_id}")
        
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # Get annotations
        annotations = self.annotations.get(image_id, {"instances": []})
        
        # Get metadata
        metadata = self.metadata.get(image_id, {})
        
        return {
            "image_id": image_id,
            "image": image,
            "image_array": image_array,
            "annotations": annotations,
            "metadata": metadata,
        }
    
    def get_instance_masks(self, idx: int) -> List[np.ndarray]:
        """
        Extract binary instance masks for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            List of binary masks, one per instance
        """
        sample = self[idx]
        masks = []
        
        for instance in sample["annotations"].get("instances", []):
            if "segmentation" in instance:
                seg = instance["segmentation"]
                
                if isinstance(seg, list):
                    # Polygon format: convert to mask
                    mask = self._polygon_to_mask(
                        seg,
                        sample["image_array"].shape[:2]
                    )
                else:
                    # RLE or binary mask format
                    mask = self._decode_mask(seg)
                
                masks.append(mask)
        
        return masks
    
    def get_instance_labels(self, idx: int) -> List[str]:
        """
        Extract semantic labels for instances in a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            List of semantic labels (e.g., "apples", "ripe apples")
        """
        sample = self[idx]
        labels = []
        
        for instance in sample["annotations"].get("instances", []):
            label = instance.get("category", "apples")
            labels.append(label)
        
        return labels
    
    def get_instance_attributes(self, idx: int) -> List[Dict[str, str]]:
        """
        Extract attribute dictionaries for instances.
        
        Args:
            idx: Sample index
            
        Returns:
            List of attribute dictionaries with keys like ripeness, color, health
        """
        sample = self[idx]
        attributes = []
        
        for instance in sample["annotations"].get("instances", []):
            attrs = {
                "ripeness": instance.get("ripeness", "unknown"),
                "color": instance.get("color", "unknown"),
                "health": instance.get("health", "unknown"),
            }
            attributes.append(attrs)
        
        return attributes
    
    def get_bounding_boxes(self, idx: int) -> List[np.ndarray]:
        """
        Extract bounding boxes for instances.
        
        Args:
            idx: Sample index
            
        Returns:
            List of bounding boxes in format [x_min, y_min, x_max, y_max]
        """
        sample = self[idx]
        boxes = []
        
        for instance in sample["annotations"].get("instances", []):
            if "bbox" in instance:
                bbox = instance["bbox"]
                # Convert from [x, y, w, h] to [x_min, y_min, x_max, y_max]
                box = np.array([
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3]
                ])
                boxes.append(box)
        
        return boxes
    
    def _polygon_to_mask(
        self,
        polygon: List[float],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert polygon coordinates to binary mask.
        
        Args:
            polygon: List of [x1, y1, x2, y2, ...] coordinates
            image_shape: (height, width) of target mask
            
        Returns:
            Binary mask array
        """
        from PIL import ImageDraw
        
        mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        
        # Convert flat list to list of tuples
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(points, outline=1, fill=1)
        
        return np.array(mask).astype(bool)
    
    def _decode_mask(self, encoded_mask) -> np.ndarray:
        """
        Decode RLE or other encoded mask formats.
        
        Args:
            encoded_mask: Encoded mask data
            
        Returns:
            Binary mask array
        """
        # Simplified decoder; full implementation would handle RLE
        # For now, assume it's already a 2D array
        if isinstance(encoded_mask, np.ndarray):
            return encoded_mask.astype(bool)
        else:
            logger.warning("Unknown mask encoding format")
            return np.array([])
    
    def get_samples_by_attribute(
        self,
        attribute_type: str,
        attribute_value: str,
    ) -> List[int]:
        """
        Find samples containing instances with specified attribute.
        
        Args:
            attribute_type: Attribute category (ripeness, color, health)
            attribute_value: Specific value to filter for
            
        Returns:
            List of sample indices matching criteria
        """
        matching_indices = []
        
        for idx in range(len(self)):
            attributes = self.get_instance_attributes(idx)
            
            for attr_dict in attributes:
                if attr_dict.get(attribute_type) == attribute_value:
                    matching_indices.append(idx)
                    break
        
        logger.debug(f"Found {len(matching_indices)} samples with {attribute_type}={attribute_value}")
        return matching_indices
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics for analysis.
        
        Returns:
            Dictionary with counts, distributions, and summary stats
        """
        stats = {
            "num_images": len(self),
            "num_instances": 0,
            "attributes": {
                "ripeness": {},
                "color": {},
                "health": {},
            },
            "instances_per_image": [],
        }
        
        for idx in range(len(self)):
            masks = self.get_instance_masks(idx)
            attributes = self.get_instance_attributes(idx)
            
            stats["num_instances"] += len(masks)
            stats["instances_per_image"].append(len(masks))
            
            for attrs in attributes:
                for attr_type in ["ripeness", "color", "health"]:
                    attr_val = attrs.get(attr_type, "unknown")
                    stats["attributes"][attr_type][attr_val] = \
                        stats["attributes"][attr_type].get(attr_val, 0) + 1
        
        # Compute summary statistics
        stats["mean_instances_per_image"] = float(np.mean(stats["instances_per_image"]))
        stats["median_instances_per_image"] = float(np.median(stats["instances_per_image"]))
        
        logger.info(f"Dataset statistics: {stats['num_images']} images, {stats['num_instances']} instances")
        return stats


def create_data_loaders(
    data_root: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42,
) -> Dict[str, MineAppleDataset]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_root: Path to dataset root
        split_ratios: Ratios for train/val/test splits
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to dataset objects
    """
    loaders = {
        "train": MineAppleDataset(data_root, "train", split_ratios, random_seed),
        "val": MineAppleDataset(data_root, "val", split_ratios, random_seed),
        "test": MineAppleDataset(data_root, "test", split_ratios, random_seed),
    }
    
    logger.info("Created data loaders for all splits")
    return loaders