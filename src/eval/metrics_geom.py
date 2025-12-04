#!/usr/bin/env python3
"""
Geometric Segmentation Metrics for SAM2 Evaluation

This module implements evaluation metrics focused on geometric accuracy
and spatial precision, which are the primary strengths of prompt-based
segmentation models like SAM2. These metrics assess mask quality without
considering semantic understanding.

Metrics include intersection-over-union, boundary accuracy, temporal
consistency, and prompt efficiency measures.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute Intersection over Union between predicted and ground truth masks.
    
    IoU is the fundamental metric for segmentation quality, measuring
    the overlap between prediction and ground truth normalized by their union.
    This is purely geometric and does not consider semantic correctness.
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
        
    Returns:
        IoU score in range [0, 1], higher is better
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


def compute_dice_coefficient(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute Dice coefficient (F1 score) for segmentation.
    
    Dice coefficient is the harmonic mean of precision and recall,
    emphasizing the importance of both false positives and false negatives.
    It is more sensitive to small objects than IoU.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dice score in range [0, 1], higher is better
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2 * intersection / (pred_sum + gt_sum)
    return float(dice)


def compute_boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
) -> float:
    """
    Compute F1 score for boundary accuracy within tolerance distance.
    
    Boundary metrics are critical for SAM2 evaluation as they measure
    the precision of mask edges, which is a key strength of prompt-based
    segmentation. Allows small spatial tolerance to account for annotation
    ambiguity.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        tolerance: Maximum distance in pixels for boundary matching
        
    Returns:
        Boundary F1 score in range [0, 1]
    """
    # Extract boundaries using morphological operations
    from scipy.ndimage import binary_erosion
    
    pred_boundary = pred_mask & ~binary_erosion(pred_mask)
    gt_boundary = gt_mask & ~binary_erosion(gt_mask)
    
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_boundary)
    gt_dist = distance_transform_edt(~gt_boundary)
    
    # Find boundary pixels within tolerance
    pred_match = (gt_dist[pred_boundary] <= tolerance).sum()
    gt_match = (pred_dist[gt_boundary] <= tolerance).sum()
    
    # Compute precision and recall
    precision = pred_match / pred_boundary.sum() if pred_boundary.sum() > 0 else 0.0
    recall = gt_match / gt_boundary.sum() if gt_boundary.sum() > 0 else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def compute_mean_iou_at_thresholds(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    thresholds: List[float] = [0.5, 0.75, 0.9, 0.95],
) -> Dict[str, float]:
    """
    Compute mean IoU and success rates at multiple thresholds.
    
    This provides a comprehensive view of segmentation quality across
    different strictness levels, revealing how performance degrades with
    higher accuracy requirements.
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        thresholds: IoU thresholds to evaluate at
        
    Returns:
        Dictionary with mIoU and success rates at each threshold
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of predictions and ground truths must match")
    
    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        iou = compute_iou(pred, gt)
        ious.append(iou)
    
    results = {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "std_iou": float(np.std(ious)),
    }
    
    # Compute success rate at each threshold
    for thresh in thresholds:
        success_rate = np.mean([iou >= thresh for iou in ious])
        results[f"success_rate_{thresh}"] = float(success_rate)
    
    logger.debug(f"Computed mIoU metrics over {len(ious)} masks")
    return results


def compute_instance_segmentation_metrics(
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute instance-level metrics including precision, recall, and AP.
    
    Instance metrics evaluate whether SAM2 detects the correct number
    of objects and segments each accurately. This is important for scenes
    with multiple objects.
    
    Args:
        pred_masks: List of predicted instance masks
        pred_scores: Confidence scores for each prediction
        gt_masks: List of ground truth instance masks
        iou_threshold: IoU threshold for matching instances
        
    Returns:
        Dictionary with precision, recall, F1, and average precision
    """
    if len(pred_masks) != len(pred_scores):
        raise ValueError("Each prediction must have a score")
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_masks = [pred_masks[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    # Match predictions to ground truths
    matched_gt = set()
    true_positives = 0
    
    for pred_mask in pred_masks:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_mask in enumerate(gt_masks):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    false_positives = len(pred_masks) - true_positives
    false_negatives = len(gt_masks) - len(matched_gt)
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_temporal_consistency(
    masks_sequence: List[np.ndarray],
) -> float:
    """
    Compute temporal consistency score for video segmentation.
    
    Temporal consistency measures how stable SAM2's predictions are across
    consecutive video frames. High jitter or flickering indicates poor
    temporal modeling.
    
    Args:
        masks_sequence: List of masks across temporal frames
        
    Returns:
        Temporal consistency score in range [0, 1], higher is better
    """
    if len(masks_sequence) < 2:
        return 1.0
    
    ious = []
    for i in range(len(masks_sequence) - 1):
        iou = compute_iou(masks_sequence[i], masks_sequence[i + 1])
        ious.append(iou)
    
    consistency = float(np.mean(ious))
    
    logger.debug(f"Computed temporal consistency over {len(masks_sequence)} frames")
    return consistency


def compute_prompt_efficiency(
    ious_by_num_clicks: Dict[int, List[float]],
) -> Dict[str, float]:
    """
    Compute metrics related to prompt efficiency and interactivity.
    
    Prompt efficiency measures how quickly SAM2 reaches high quality
    with increasing interactive refinement. Lower clicks needed indicates
    better prompt understanding.
    
    Args:
        ious_by_num_clicks: Dictionary mapping number of clicks to IoU scores
        
    Returns:
        Dictionary with efficiency metrics
    """
    metrics = {}
    
    # Find average clicks needed to reach quality thresholds
    thresholds = [0.8, 0.85, 0.9]
    for thresh in thresholds:
        clicks_needed = []
        for num_clicks, ious in sorted(ious_by_num_clicks.items()):
            if np.mean(ious) >= thresh:
                clicks_needed.append(num_clicks)
                break
        
        if clicks_needed:
            metrics[f"clicks_to_{thresh}_iou"] = float(clicks_needed[0])
        else:
            metrics[f"clicks_to_{thresh}_iou"] = float('inf')
    
    # Compute area under IoU curve
    if ious_by_num_clicks:
        max_clicks = max(ious_by_num_clicks.keys())
        auc = 0.0
        for num_clicks in range(1, max_clicks + 1):
            if num_clicks in ious_by_num_clicks:
                auc += np.mean(ious_by_num_clicks[num_clicks])
        auc /= max_clicks
        metrics["auc_iou"] = float(auc)
    
    logger.debug("Computed prompt efficiency metrics")
    return metrics


def compute_mask_quality_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute comprehensive mask quality score combining multiple factors.
    
    This aggregates IoU, boundary accuracy, and shape similarity into
    a single quality score for ranking segmentations.
    
    Args:
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        
    Returns:
        Quality score in range [0, 1]
    """
    # Compute individual components
    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice_coefficient(pred_mask, gt_mask)
    boundary_f1 = compute_boundary_f1(pred_mask, gt_mask, tolerance=2)
    
    # Weighted combination
    quality = 0.5 * iou + 0.3 * boundary_f1 + 0.2 * dice
    
    return float(quality)


def compute_size_stratified_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    size_bins: List[int] = [32*32, 96*96],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by object size.
    
    Performance often varies significantly by object size, with small objects
    being particularly challenging. This stratification reveals size-dependent
    behaviors.
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        size_bins: Area thresholds defining small, medium, large objects
        
    Returns:
        Dictionary mapping size categories to their metrics
    """
    size_categories = {
        "small": [],
        "medium": [],
        "large": [],
    }
    
    for pred, gt in zip(pred_masks, gt_masks):
        area = gt.sum()
        
        if area < size_bins[0]:
            category = "small"
        elif area < size_bins[1]:
            category = "medium"
        else:
            category = "large"
        
        iou = compute_iou(pred, gt)
        size_categories[category].append(iou)
    
    results = {}
    for category, ious in size_categories.items():
        if ious:
            results[category] = {
                "count": len(ious),
                "mean_iou": float(np.mean(ious)),
                "median_iou": float(np.median(ious)),
            }
        else:
            results[category] = {
                "count": 0,
                "mean_iou": 0.0,
                "median_iou": 0.0,
            }
    
    logger.info(f"Computed size-stratified metrics for {len(pred_masks)} masks")
    return results


def compute_occlusion_robustness(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    occlusion_levels: List[float],
) -> Dict[float, float]:
    """
    Compute IoU under simulated occlusion to test robustness.
    
    Occluded objects are common in real scenarios. This evaluates how
    well SAM2 handles partial visibility.
    
    Args:
        pred_masks: Predicted masks
        gt_masks: Ground truth masks
        occlusion_levels: List of occlusion ratios to simulate (0.0 to 1.0)
        
    Returns:
        Dictionary mapping occlusion levels to mean IoU
    """
    results = {}
    
    for occlusion in occlusion_levels:
        occluded_ious = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            # Simulate occlusion by randomly removing pixels
            occluded_pred = pred.copy()
            mask_pixels = np.where(occluded_pred)
            num_occlude = int(len(mask_pixels[0]) * occlusion)
            
            if num_occlude > 0:
                occlude_indices = np.random.choice(len(mask_pixels[0]), num_occlude, replace=False)
                occluded_pred[mask_pixels[0][occlude_indices], mask_pixels[1][occlude_indices]] = False
            
            iou = compute_iou(occluded_pred, gt)
            occluded_ious.append(iou)
        
        results[occlusion] = float(np.mean(occluded_ious))
    
    logger.debug(f"Computed occlusion robustness at {len(occlusion_levels)} levels")
    return results


def compute_multi_object_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for multi-object segmentation scenes.
    
    Evaluates SAM2's ability to handle multiple objects simultaneously,
    including proper instance separation and completeness.
    
    Args:
        pred_masks: 3D array (N, H, W) with N predicted instance masks
        gt_masks: 3D array (M, H, W) with M ground truth instance masks
        
    Returns:
        Dictionary with multi-object metrics
    """
    from scipy.optimize import linear_sum_assignment
    
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    # Build IoU matrix
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = compute_iou(pred_masks[i], gt_masks[j])
    
    # Optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    matched_ious = iou_matrix[row_ind, col_ind]
    
    return {
        "num_predicted": num_pred,
        "num_ground_truth": num_gt,
        "num_matched": len(matched_ious),
        "mean_matched_iou": float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0,
        "precision": len(matched_ious) / num_pred if num_pred > 0 else 0.0,
        "recall": len(matched_ious) / num_gt if num_gt > 0 else 0.0,
    }


def aggregate_geometric_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive geometric metric suite for SAM2 evaluation.
    
    This aggregates all relevant geometric metrics into a single report
    for complete performance assessment.
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        pred_scores: Optional confidence scores for predictions
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Basic overlap metrics
    metrics.update(compute_mean_iou_at_thresholds(pred_masks, gt_masks))
    
    # Boundary accuracy
    boundary_f1_scores = [compute_boundary_f1(p, g) for p, g in zip(pred_masks, gt_masks)]
    metrics["mean_boundary_f1"] = float(np.mean(boundary_f1_scores))
    
    # Dice coefficients
    dice_scores = [compute_dice_coefficient(p, g) for p, g in zip(pred_masks, gt_masks)]
    metrics["mean_dice"] = float(np.mean(dice_scores))
    
    # Instance-level metrics if scores provided
    if pred_scores is not None:
        instance_metrics = compute_instance_segmentation_metrics(
            pred_masks, pred_scores, gt_masks
        )
        metrics.update(instance_metrics)
    
    # Size-stratified analysis
    size_metrics = compute_size_stratified_metrics(pred_masks, gt_masks)
    for size_cat, size_results in size_metrics.items():
        for metric_name, value in size_results.items():
            metrics[f"{size_cat}_{metric_name}"] = value
    
    logger.info(f"Aggregated geometric metrics for {len(pred_masks)} masks")
    return metrics
