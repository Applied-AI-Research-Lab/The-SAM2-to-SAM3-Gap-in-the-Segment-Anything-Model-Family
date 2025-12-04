#!/usr/bin/env python3
"""
Concept-Level Segmentation Metrics for SAM3 Evaluation

This module implements evaluation metrics focused on semantic understanding
and concept reasoning, which are the key capabilities of vision-language
models like SAM3. These metrics assess whether the model segments the
correct semantic concepts, not just geometrically accurate regions.

Metrics include concept recall, semantic grounding accuracy, open-vocabulary
performance, and attribute understanding measures.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_concept_recall(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    gt_labels: List[str],
    pred_labels: List[str],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute recall for each concept category.
    
    Concept recall measures whether SAM3 detects all instances of a given
    semantic concept. Unlike geometric IoU, this requires matching both
    spatial extent and semantic label.
    
    Args:
        pred_masks: List of predicted instance masks
        gt_masks: List of ground truth instance masks
        gt_labels: Semantic labels for ground truth instances
        pred_labels: Semantic labels for predictions
        iou_threshold: Minimum IoU for spatial matching
        
    Returns:
        Dictionary mapping each concept to its recall score
    """
    # Get unique concepts
    unique_concepts = set(gt_labels)
    
    concept_recalls = {}
    
    for concept in unique_concepts:
        # Get ground truth instances of this concept
        gt_concept_indices = [i for i, label in enumerate(gt_labels) if label == concept]
        gt_concept_masks = [gt_masks[i] for i in gt_concept_indices]
        
        # Get predictions for this concept
        pred_concept_indices = [i for i, label in enumerate(pred_labels) if label == concept]
        pred_concept_masks = [pred_masks[i] for i in pred_concept_indices]
        
        if len(gt_concept_masks) == 0:
            continue
        
        # Count how many ground truth instances are detected
        detected = 0
        for gt_mask in gt_concept_masks:
            for pred_mask in pred_concept_masks:
                from .metrics_geom import compute_iou
                iou = compute_iou(pred_mask, gt_mask)
                if iou >= iou_threshold:
                    detected += 1
                    break
        
        recall = detected / len(gt_concept_masks)
        concept_recalls[concept] = float(recall)
    
    # Compute mean across all concepts
    if concept_recalls:
        concept_recalls["mean_concept_recall"] = float(np.mean(list(concept_recalls.values())))
    
    logger.info(f"Computed concept recall for {len(unique_concepts)} concepts")
    return concept_recalls


def compute_concept_precision(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    gt_labels: List[str],
    pred_labels: List[str],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision for each concept category.
    
    Concept precision measures what fraction of predictions for a concept
    are correct. This reveals whether SAM3 hallucinates concepts or
    confuses similar categories.
    
    Args:
        pred_masks: Predicted instance masks
        gt_masks: Ground truth instance masks
        gt_labels: Ground truth semantic labels
        pred_labels: Predicted semantic labels
        iou_threshold: Minimum IoU for spatial matching
        
    Returns:
        Dictionary mapping concepts to precision scores
    """
    unique_concepts = set(pred_labels)
    
    concept_precisions = {}
    
    for concept in unique_concepts:
        pred_concept_indices = [i for i, label in enumerate(pred_labels) if label == concept]
        pred_concept_masks = [pred_masks[i] for i in pred_concept_indices]
        
        gt_concept_indices = [i for i, label in enumerate(gt_labels) if label == concept]
        gt_concept_masks = [gt_masks[i] for i in gt_concept_indices]
        
        if len(pred_concept_masks) == 0:
            continue
        
        # Count how many predictions are correct
        correct = 0
        for pred_mask in pred_concept_masks:
            for gt_mask in gt_concept_masks:
                from .metrics_geom import compute_iou
                iou = compute_iou(pred_mask, gt_mask)
                if iou >= iou_threshold:
                    correct += 1
                    break
        
        precision = correct / len(pred_concept_masks)
        concept_precisions[concept] = float(precision)
    
    if concept_precisions:
        concept_precisions["mean_concept_precision"] = float(np.mean(list(concept_precisions.values())))
    
    logger.info(f"Computed concept precision for {len(unique_concepts)} concepts")
    return concept_precisions


def compute_semantic_grounding_accuracy(
    pred_masks: List[np.ndarray],
    text_prompts: List[str],
    gt_labels: List[str],
    label_similarity_fn: Optional[callable] = None,
) -> float:
    """
    Compute accuracy of semantic grounding between text and masks.
    
    This measures whether SAM3 correctly interprets text prompts and
    grounds them to appropriate visual regions. It assesses the vision-
    language alignment quality.
    
    Args:
        pred_masks: Predicted masks
        text_prompts: Text prompts used for each prediction
        gt_labels: Ground truth semantic labels
        label_similarity_fn: Optional function to compute semantic similarity
                            between prompt and label (default: exact match)
        
    Returns:
        Grounding accuracy in range [0, 1]
    """
    if len(pred_masks) == 0:
        return 0.0
    
    if label_similarity_fn is None:
        # Default: exact string match after normalization
        def label_similarity_fn(prompt, label):
            prompt_norm = prompt.lower().strip()
            label_norm = label.lower().strip()
            return 1.0 if prompt_norm == label_norm or label_norm in prompt_norm else 0.0
    
    correct_groundings = 0
    
    for prompt, pred_mask, gt_label in zip(text_prompts, pred_masks, gt_labels):
        similarity = label_similarity_fn(prompt, gt_label)
        if similarity > 0.5:
            correct_groundings += 1
    
    accuracy = correct_groundings / len(pred_masks)
    
    logger.debug(f"Semantic grounding accuracy: {accuracy:.3f}")
    return float(accuracy)


def compute_attribute_accuracy(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_attributes: List[Dict[str, str]],
    gt_attributes: List[Dict[str, str]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute accuracy of attribute prediction for matched instances.
    
    This evaluates SAM3's ability to understand fine-grained semantic
    attributes like color, state, or quality. Requires both spatial
    matching and attribute correctness.
    
    Args:
        pred_masks: Predicted instance masks
        gt_masks: Ground truth masks
        pred_attributes: Predicted attribute dictionaries for each instance
        gt_attributes: Ground truth attribute dictionaries
        iou_threshold: Minimum IoU for spatial matching
        
    Returns:
        Dictionary with per-attribute accuracy scores
    """
    from .metrics_geom import compute_iou
    
    # Match predictions to ground truths spatially
    matches = []
    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_mask in enumerate(gt_masks):
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matches.append((i, best_gt_idx))
    
    if len(matches) == 0:
        logger.warning("No spatial matches found for attribute evaluation")
        return {}
    
    # Collect all attribute types
    all_attr_types = set()
    for attrs in gt_attributes:
        all_attr_types.update(attrs.keys())
    
    # Compute accuracy per attribute type
    attribute_accuracies = {}
    
    for attr_type in all_attr_types:
        correct = 0
        total = 0
        
        for pred_idx, gt_idx in matches:
            if attr_type in gt_attributes[gt_idx]:
                total += 1
                
                gt_value = gt_attributes[gt_idx][attr_type]
                pred_value = pred_attributes[pred_idx].get(attr_type, None)
                
                if pred_value == gt_value:
                    correct += 1
        
        if total > 0:
            attribute_accuracies[attr_type] = float(correct / total)
    
    if attribute_accuracies:
        attribute_accuracies["mean_attribute_accuracy"] = float(np.mean(list(attribute_accuracies.values())))
    
    logger.info(f"Computed attribute accuracy for {len(attribute_accuracies)-1} attributes")
    return attribute_accuracies


def compute_open_vocabulary_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    seen_concepts: Set[str],
    unseen_concepts: Set[str],
    gt_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, float]:
    """
    Compute metrics separately for seen vs unseen concepts.
    
    Open-vocabulary capability is a key advantage of SAM3 over SAM2.
    This measures performance on novel concepts not seen during training,
    which requires true semantic understanding.
    
    Args:
        pred_masks: Predicted masks
        gt_masks: Ground truth masks
        seen_concepts: Set of concepts from training data
        unseen_concepts: Set of novel concepts
        gt_labels: Ground truth labels
        pred_labels: Predicted labels
        
    Returns:
        Dictionary with seen and unseen performance breakdown
    """
    from .metrics_geom import compute_iou
    
    results = {
        "seen": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
        "unseen": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
    }
    
    # Categorize ground truths
    for i, (gt_mask, gt_label) in enumerate(zip(gt_masks, gt_labels)):
        category = "seen" if gt_label in seen_concepts else "unseen"
        
        # Find best matching prediction
        best_iou = 0.0
        best_pred_idx = -1
        
        for j, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
            if pred_label == gt_label:
                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
        
        if best_iou >= 0.5:
            results[category]["tp"] += 1
            results[category]["ious"].append(best_iou)
        else:
            results[category]["fn"] += 1
    
    # Count false positives
    for pred_label in pred_labels:
        category = "seen" if pred_label in seen_concepts else "unseen"
        # Simplified FP counting (full implementation would check unmatched predictions)
    
    # Compute metrics for each category
    metrics = {}
    for category in ["seen", "unseen"]:
        tp = results[category]["tp"]
        fp = results[category]["fp"]
        fn = results[category]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(results[category]["ious"])) if results[category]["ious"] else 0.0
        
        metrics[f"{category}_precision"] = float(precision)
        metrics[f"{category}_recall"] = float(recall)
        metrics[f"{category}_f1"] = float(f1)
        metrics[f"{category}_mean_iou"] = mean_iou
    
    # Compute generalization gap
    if metrics["seen_f1"] > 0:
        metrics["generalization_gap"] = float(metrics["seen_f1"] - metrics["unseen_f1"])
    
    logger.info(f"Computed open-vocabulary metrics for {len(seen_concepts)} seen and {len(unseen_concepts)} unseen concepts")
    return metrics


def compute_compositional_understanding(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    compositional_prompts: List[Tuple[str, Dict[str, str]]],
    gt_attributes: List[Dict[str, str]],
) -> float:
    """
    Evaluate ability to understand compositional concept descriptions.
    
    Compositional understanding tests whether SAM3 correctly interprets
    prompts combining multiple attributes (e.g., "ripe red apples"),
    requiring semantic composition.
    
    Args:
        pred_masks: Predicted masks
        gt_masks: Ground truth masks
        compositional_prompts: List of (prompt_text, attribute_dict) tuples
        gt_attributes: Ground truth attribute dictionaries
        
    Returns:
        Compositional accuracy score
    """
    from .metrics_geom import compute_iou
    
    correct = 0
    total = len(compositional_prompts)
    
    for i, (prompt_text, prompt_attrs) in enumerate(compositional_prompts):
        if i >= len(pred_masks):
            break
        
        pred_mask = pred_masks[i]
        
        # Find ground truth with matching attributes
        best_iou = 0.0
        best_match = False
        
        for gt_mask, gt_attrs in zip(gt_masks, gt_attributes):
            # Check if attributes match
            attrs_match = all(
                gt_attrs.get(k) == v for k, v in prompt_attrs.items()
            )
            
            if attrs_match:
                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    if iou >= 0.5:
                        best_match = True
        
        if best_match:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    logger.debug(f"Compositional understanding: {accuracy:.3f}")
    return float(accuracy)


def compute_negative_prompt_effectiveness(
    pred_with_negative: List[np.ndarray],
    pred_without_negative: List[np.ndarray],
    gt_positive_masks: List[np.ndarray],
    gt_negative_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate how well negative prompts improve segmentation precision.
    
    Negative prompts should help SAM3 exclude similar but incorrect
    concepts. This measures the improvement from using negatives.
    
    Args:
        pred_with_negative: Predictions using negative prompts
        pred_without_negative: Predictions without negatives (baseline)
        gt_positive_masks: Ground truth for target concept
        gt_negative_masks: Ground truth for negative concept
        
    Returns:
        Dictionary with negative prompt effectiveness metrics
    """
    from .metrics_geom import compute_iou
    
    # Compute false positive rates
    fp_with_neg = 0
    fp_without_neg = 0
    
    for pred_with, pred_without in zip(pred_with_negative, pred_without_negative):
        # Check overlap with negative masks
        for neg_mask in gt_negative_masks:
            overlap_with = compute_iou(pred_with, neg_mask)
            overlap_without = compute_iou(pred_without, neg_mask)
            
            if overlap_with > 0.3:
                fp_with_neg += 1
            if overlap_without > 0.3:
                fp_without_neg += 1
    
    # Compute positive recall
    tp_with_neg = 0
    tp_without_neg = 0
    
    for pred_with, pred_without in zip(pred_with_negative, pred_without_negative):
        for pos_mask in gt_positive_masks:
            iou_with = compute_iou(pred_with, pos_mask)
            iou_without = compute_iou(pred_without, pos_mask)
            
            if iou_with >= 0.5:
                tp_with_neg += 1
            if iou_without >= 0.5:
                tp_without_neg += 1
    
    metrics = {
        "fp_reduction": float((fp_without_neg - fp_with_neg) / fp_without_neg) if fp_without_neg > 0 else 0.0,
        "recall_with_negative": float(tp_with_neg / len(gt_positive_masks)) if len(gt_positive_masks) > 0 else 0.0,
        "recall_without_negative": float(tp_without_neg / len(gt_positive_masks)) if len(gt_positive_masks) > 0 else 0.0,
    }
    
    logger.debug("Computed negative prompt effectiveness")
    return metrics


def compute_concept_confusion_matrix(
    pred_labels: List[str],
    gt_labels: List[str],
    concepts: List[str],
) -> np.ndarray:
    """
    Compute confusion matrix for concept classification.
    
    Reveals which concepts SAM3 confuses with each other, providing
    insight into semantic understanding limitations.
    
    Args:
        pred_labels: Predicted concept labels
        gt_labels: Ground truth concept labels
        concepts: Ordered list of concept names
        
    Returns:
        Confusion matrix as 2D array (ground_truth x predicted)
    """
    n_concepts = len(concepts)
    confusion = np.zeros((n_concepts, n_concepts), dtype=int)
    
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    
    for pred, gt in zip(pred_labels, gt_labels):
        if pred in concept_to_idx and gt in concept_to_idx:
            gt_idx = concept_to_idx[gt]
            pred_idx = concept_to_idx[pred]
            confusion[gt_idx, pred_idx] += 1
    
    logger.info(f"Computed confusion matrix for {n_concepts} concepts")
    return confusion


def compute_semantic_consistency(
    masks_for_synonyms: Dict[str, List[np.ndarray]],
) -> float:
    """
    Compute consistency of segmentation across synonym prompts.
    
    Different phrasings of the same concept should produce similar
    segmentations. This measures robustness to linguistic variation.
    
    Args:
        masks_for_synonyms: Dictionary mapping synonym phrases to their masks
        
    Returns:
        Consistency score (mean IoU across synonym pairs)
    """
    from .metrics_geom import compute_iou
    
    synonym_groups = list(masks_for_synonyms.values())
    
    if len(synonym_groups) < 2:
        return 1.0
    
    ious = []
    for i in range(len(synonym_groups)):
        for j in range(i + 1, len(synonym_groups)):
            masks_i = synonym_groups[i]
            masks_j = synonym_groups[j]
            
            # Compare masks pairwise
            for mask_i in masks_i:
                for mask_j in masks_j:
                    iou = compute_iou(mask_i, mask_j)
                    ious.append(iou)
    
    consistency = float(np.mean(ious)) if ious else 0.0
    
    logger.debug(f"Semantic consistency across synonyms: {consistency:.3f}")
    return consistency


def aggregate_concept_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_labels: List[str],
    gt_labels: List[str],
    pred_attributes: Optional[List[Dict[str, str]]] = None,
    gt_attributes: Optional[List[Dict[str, str]]] = None,
    text_prompts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive concept-level metric suite for SAM3 evaluation.
    
    This aggregates all semantic metrics into a single report for
    complete concept understanding assessment.
    
    Args:
        pred_masks: Predicted masks
        gt_masks: Ground truth masks
        pred_labels: Predicted semantic labels
        gt_labels: Ground truth semantic labels
        pred_attributes: Optional predicted attributes
        gt_attributes: Optional ground truth attributes
        text_prompts: Optional text prompts used
        
    Returns:
        Dictionary with all computed concept metrics
    """
    metrics = {}
    
    # Core concept metrics
    concept_recall = compute_concept_recall(pred_masks, gt_masks, gt_labels, pred_labels)
    metrics.update(concept_recall)
    
    concept_precision = compute_concept_precision(pred_masks, gt_masks, gt_labels, pred_labels)
    metrics.update(concept_precision)
    
    # Semantic grounding
    if text_prompts is not None:
        grounding_acc = compute_semantic_grounding_accuracy(pred_masks, text_prompts, gt_labels)
        metrics["semantic_grounding_accuracy"] = grounding_acc
    
    # Attribute understanding
    if pred_attributes is not None and gt_attributes is not None:
        attr_acc = compute_attribute_accuracy(pred_masks, gt_masks, pred_attributes, gt_attributes)
        metrics.update(attr_acc)
    
    logger.info(f"Aggregated concept metrics for {len(pred_masks)} predictions")
    return metrics
