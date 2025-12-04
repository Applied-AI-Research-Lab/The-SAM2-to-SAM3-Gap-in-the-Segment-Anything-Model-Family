#!/usr/bin/env python3
"""
SAM2 Prompt Generation Utilities

This module provides utilities for generating geometric prompts for SAM2,
including point coordinates, bounding boxes, and grid-based automatic
generation strategies. These prompts represent purely spatial information
without semantic understanding.

SAM2 relies on explicit visual cues to define segmentation targets, making
prompt quality critical for performance. This module helps generate diverse
and effective prompt sets for evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_point_prompt(
    mask: np.ndarray,
    num_points: int = 1,
    strategy: str = "centroid",
    foreground_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point prompts from ground truth masks for SAM2.
    
    Point prompts are the most basic form of spatial guidance for SAM2.
    This function extracts representative points from masks to simulate
    interactive annotation scenarios.
    
    Args:
        mask: Binary mask (H, W) with object pixels marked as True
        num_points: Number of points to generate
        strategy: Point selection strategy:
                 - centroid: Use mask center of mass
                 - random: Random points within mask
                 - boundary: Points near mask boundaries
                 - grid: Uniformly spaced grid within mask
        foreground_only: If True, only generate positive (foreground) points
        
    Returns:
        Tuple of (coordinates, labels):
            - coordinates: Array of shape (N, 2) with (x, y) positions
            - labels: Array of shape (N,) with 1 for foreground, 0 for background
    """
    if mask.sum() == 0:
        logger.warning("Empty mask provided, returning empty point set")
        return np.array([]), np.array([])
    
    coords = []
    labels = []
    
    # Get foreground pixel locations
    ys, xs = np.where(mask)
    
    if strategy == "centroid":
        # Use center of mass as point
        centroid_y = int(np.mean(ys))
        centroid_x = int(np.mean(xs))
        coords.append([centroid_x, centroid_y])
        labels.append(1)
        
        # Add additional random points if requested
        if num_points > 1:
            indices = np.random.choice(len(xs), size=num_points-1, replace=False)
            for idx in indices:
                coords.append([xs[idx], ys[idx]])
                labels.append(1)
    
    elif strategy == "random":
        # Select random foreground points
        num_fg = num_points if foreground_only else max(1, num_points // 2)
        indices = np.random.choice(len(xs), size=num_fg, replace=False)
        
        for idx in indices:
            coords.append([xs[idx], ys[idx]])
            labels.append(1)
        
        # Add background points if needed
        if not foreground_only:
            num_bg = num_points - num_fg
            bg_ys, bg_xs = np.where(~mask)
            if len(bg_xs) > 0:
                bg_indices = np.random.choice(len(bg_xs), size=num_bg, replace=False)
                for idx in bg_indices:
                    coords.append([bg_xs[idx], bg_ys[idx]])
                    labels.append(0)
    
    elif strategy == "boundary":
        # Find boundary pixels using erosion
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        
        bound_ys, bound_xs = np.where(boundary)
        if len(bound_xs) < num_points:
            logger.warning(f"Only {len(bound_xs)} boundary points available, using all")
            indices = np.arange(len(bound_xs))
        else:
            indices = np.random.choice(len(bound_xs), size=num_points, replace=False)
        
        for idx in indices:
            coords.append([bound_xs[idx], bound_ys[idx]])
            labels.append(1)
    
    elif strategy == "grid":
        # Create uniform grid within mask bounds
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        
        grid_size = int(np.sqrt(num_points))
        y_points = np.linspace(min_y, max_y, grid_size)
        x_points = np.linspace(min_x, max_x, grid_size)
        
        for y in y_points:
            for x in x_points:
                y_int, x_int = int(y), int(x)
                if mask[y_int, x_int]:
                    coords.append([x_int, y_int])
                    labels.append(1)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    coords = np.array(coords)
    labels = np.array(labels)
    
    logger.debug(f"Generated {len(coords)} points using {strategy} strategy")
    return coords, labels


def generate_box_prompt(
    mask: np.ndarray,
    padding: int = 0,
    jitter: float = 0.0,
) -> np.ndarray:
    """
    Generate bounding box prompt from ground truth mask.
    
    Box prompts provide coarse spatial guidance by defining rectangular
    regions. This is less precise than points but covers the full object extent.
    
    Args:
        mask: Binary mask (H, W)
        padding: Pixels to add around tight bounding box
        jitter: Amount of random noise to add to box coordinates (0.0-1.0)
                Simulates imperfect human annotations
        
    Returns:
        Box coordinates as array [x_min, y_min, x_max, y_max]
    """
    if mask.sum() == 0:
        logger.warning("Empty mask, returning zero box")
        return np.array([0, 0, 0, 0])
    
    ys, xs = np.where(mask)
    
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(mask.shape[1] - 1, x_max + padding)
    y_max = min(mask.shape[0] - 1, y_max + padding)
    
    # Apply jitter to simulate annotation noise
    if jitter > 0:
        width = x_max - x_min
        height = y_max - y_min
        
        x_min += int(np.random.uniform(-jitter * width, jitter * width))
        x_max += int(np.random.uniform(-jitter * width, jitter * width))
        y_min += int(np.random.uniform(-jitter * height, jitter * height))
        y_max += int(np.random.uniform(-jitter * height, jitter * height))
        
        # Ensure box remains valid
        x_min = max(0, min(x_min, mask.shape[1] - 2))
        x_max = max(x_min + 1, min(x_max, mask.shape[1] - 1))
        y_min = max(0, min(y_min, mask.shape[0] - 2))
        y_max = max(y_min + 1, min(y_max, mask.shape[0] - 1))
    
    box = np.array([x_min, y_min, x_max, y_max])
    
    logger.debug(f"Generated box prompt: {box}")
    return box


def generate_mask_prompt(
    mask: np.ndarray,
    noise_level: float = 0.0,
    erosion_iterations: int = 0,
) -> np.ndarray:
    """
    Generate mask prompt by perturbing ground truth mask.
    
    Mask prompts allow iterative refinement in SAM2 by providing a coarse
    initial segmentation. This simulates scenarios where a rough mask exists
    and needs improvement.
    
    Args:
        mask: Ground truth binary mask (H, W)
        noise_level: Fraction of pixels to randomly flip (0.0-1.0)
        erosion_iterations: Number of erosion steps to shrink mask
        
    Returns:
        Perturbed mask as float array (H, W) with values in [0, 1]
    """
    prompt_mask = mask.astype(np.float32).copy()
    
    # Apply erosion to shrink mask
    if erosion_iterations > 0:
        from scipy.ndimage import binary_erosion
        prompt_mask = binary_erosion(
            prompt_mask.astype(bool),
            iterations=erosion_iterations
        ).astype(np.float32)
    
    # Add random noise
    if noise_level > 0:
        noise = np.random.random(mask.shape) < noise_level
        prompt_mask = np.logical_xor(prompt_mask, noise).astype(np.float32)
    
    logger.debug(f"Generated mask prompt with {noise_level} noise and {erosion_iterations} erosions")
    return prompt_mask


def generate_grid_prompts(
    image_size: Tuple[int, int],
    points_per_side: int = 32,
    points_per_batch: int = 64,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate grid of point prompts for automatic mask generation in SAM2.
    
    This creates a uniform spatial grid across the image, allowing SAM2 to
    segment all objects without manual prompts. Each grid point serves as
    a potential object center.
    
    Args:
        image_size: Tuple of (height, width)
        points_per_side: Number of points along each dimension
        points_per_batch: Maximum points to process together
        
    Returns:
        List of prompt dictionaries, each with coordinates and labels
    """
    height, width = image_size
    
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, points_per_side, dtype=int)
    x_coords = np.linspace(0, width - 1, points_per_side, dtype=int)
    
    grid_y, grid_x = np.meshgrid(y_coords, x_coords)
    grid_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    
    # All points are foreground prompts
    grid_labels = np.ones(len(grid_coords), dtype=int)
    
    # Split into batches
    prompts = []
    for i in range(0, len(grid_coords), points_per_batch):
        batch_coords = grid_coords[i:i + points_per_batch]
        batch_labels = grid_labels[i:i + points_per_batch]
        
        prompts.append({
            "coords": batch_coords,
            "labels": batch_labels,
        })
    
    logger.info(f"Generated {len(grid_coords)} grid points in {len(prompts)} batches")
    return prompts


def generate_multiscale_points(
    mask: np.ndarray,
    scales: List[int] = [1, 2, 4],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point prompts at multiple scales for hierarchical segmentation.
    
    This creates points representing different levels of detail, useful for
    objects with complex structures that benefit from multi-scale guidance.
    
    Args:
        mask: Binary ground truth mask
        scales: List of downsampling factors for each scale
        
    Returns:
        Tuple of (coordinates, labels) with points from all scales
    """
    all_coords = []
    all_labels = []
    
    for scale in scales:
        if scale > 1:
            # Downsample mask
            from scipy.ndimage import zoom
            scaled_mask = zoom(mask.astype(float), 1.0 / scale, order=0) > 0.5
        else:
            scaled_mask = mask
        
        # Generate points at this scale
        coords, labels = generate_point_prompt(
            scaled_mask,
            num_points=max(1, 3 // scale),
            strategy="random"
        )
        
        # Scale coordinates back to original size
        if scale > 1 and len(coords) > 0:
            coords = coords * scale
        
        all_coords.append(coords)
        all_labels.append(labels)
    
    # Combine all scales
    if all_coords:
        combined_coords = np.vstack([c for c in all_coords if len(c) > 0])
        combined_labels = np.concatenate([l for l in all_labels if len(l) > 0])
    else:
        combined_coords = np.array([])
        combined_labels = np.array([])
    
    logger.debug(f"Generated {len(combined_coords)} multi-scale points across {len(scales)} scales")
    return combined_coords, combined_labels


def generate_adaptive_grid(
    mask: np.ndarray,
    target_points: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adaptive point grid with higher density in complex regions.
    
    Instead of uniform spacing, this places more points in areas with high
    boundary complexity, providing better guidance where needed.
    
    Args:
        mask: Binary mask to analyze
        target_points: Approximate number of points to generate
        
    Returns:
        Tuple of (coordinates, labels)
    """
    if mask.sum() == 0:
        return np.array([]), np.array([])
    
    # Compute edge density map
    from scipy.ndimage import sobel
    edges = np.hypot(sobel(mask.astype(float), 0), sobel(mask.astype(float), 1))
    
    # Normalize to probability distribution
    edges_in_mask = edges * mask
    if edges_in_mask.sum() > 0:
        prob_map = edges_in_mask / edges_in_mask.sum()
    else:
        prob_map = mask.astype(float) / mask.sum()
    
    # Sample points according to probability
    ys, xs = np.where(mask)
    flat_probs = prob_map[ys, xs]
    
    num_samples = min(target_points, len(xs))
    indices = np.random.choice(
        len(xs),
        size=num_samples,
        replace=False,
        p=flat_probs / flat_probs.sum()
    )
    
    coords = np.stack([xs[indices], ys[indices]], axis=1)
    labels = np.ones(len(coords), dtype=int)
    
    logger.debug(f"Generated {len(coords)} adaptive grid points")
    return coords, labels


def simulate_interactive_clicks(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    max_clicks: int = 10,
    stop_iou: float = 0.9,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate interactive refinement clicks based on prediction errors.
    
    This generates a sequence of corrective prompts that a user would provide
    to iteratively improve a poor segmentation, measuring SAM2's ability to
    incorporate interactive feedback.
    
    Args:
        predicted_mask: Current prediction from model
        ground_truth_mask: Target correct mask
        max_clicks: Maximum number of correction clicks
        stop_iou: IoU threshold to stop clicking
        
    Returns:
        List of (coordinates, labels) tuples for each refinement iteration
    """
    click_history = []
    current_pred = predicted_mask.copy()
    
    for click_idx in range(max_clicks):
        # Compute current IoU
        intersection = (current_pred & ground_truth_mask).sum()
        union = (current_pred | ground_truth_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        
        if iou >= stop_iou:
            logger.info(f"Reached target IoU {iou:.3f} after {click_idx} clicks")
            break
        
        # Find false negatives (missed regions) and false positives (over-segmented)
        false_neg = ground_truth_mask & ~current_pred
        false_pos = current_pred & ~ground_truth_mask
        
        # Prioritize largest error
        if false_neg.sum() > false_pos.sum():
            # Add positive click on missed region
            error_ys, error_xs = np.where(false_neg)
            click_idx_rand = np.random.randint(len(error_xs))
            coord = np.array([[error_xs[click_idx_rand], error_ys[click_idx_rand]]])
            label = np.array([1])
        else:
            # Add negative click on false positive
            error_ys, error_xs = np.where(false_pos)
            click_idx_rand = np.random.randint(len(error_xs))
            coord = np.array([[error_xs[click_idx_rand], error_ys[click_idx_rand]]])
            label = np.array([0])
        
        click_history.append((coord, label))
        
        # Update prediction (simplified simulation)
        # In practice, you would re-run SAM2 with accumulated clicks
        if label[0] == 1:
            current_pred[coord[0, 1], coord[0, 0]] = True
        else:
            current_pred[coord[0, 1], coord[0, 0]] = False
    
    logger.info(f"Simulated {len(click_history)} interactive clicks")
    return click_history
