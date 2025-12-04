#!/usr/bin/env python3
"""
Main Experiment Runner for SAM2-to-SAM3 Gap Analysis

This script orchestrates complete experimental workflows comparing SAM2 and SAM3
on the MineApple dataset. It handles configuration loading, model initialization,
data processing, evaluation metric computation, and results visualization.

Usage:
    python run_experiment.py --config configs/sam2_mineapple.yml
    python run_experiment.py --config configs/sam3_mineapple.yml
    python run_experiment.py --mode compare --sam2-config configs/sam2_mineapple.yml --sam3-config configs/sam3_mineapple.yml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from models.sam2_wrapper import SAM2Wrapper
from models.sam3_wrapper import SAM3Wrapper
from prompts.sam2_prompts import generate_point_prompt, generate_box_prompt
from prompts.sam3_text_prompts import create_mineapple_prompts
from eval.metrics_geom import aggregate_geometric_metrics
from eval.metrics_concept import aggregate_concept_metrics
from utils.dataset_loader import create_data_loaders
from utils.visualization import (
    visualize_sam2_vs_sam3_comparison,
    plot_metric_comparison,
    plot_iou_distribution,
)
from utils.logging_utils import setup_logging, ExperimentTracker

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def run_sam2_experiment(
    config: Dict,
    data_loaders: Dict,
    tracker: ExperimentTracker,
) -> Dict:
    """
    Run SAM2 experiment with prompt-based segmentation.
    
    This function evaluates SAM2 using geometric prompts (points, boxes)
    and measures performance using spatial accuracy metrics.
    
    Args:
        config: Experiment configuration
        data_loaders: Dictionary of dataset loaders
        tracker: Experiment tracker for logging
        
    Returns:
        Dictionary with SAM2 results and metrics
    """
    logger.info("Starting SAM2 experiment")
    
    # Initialize SAM2 model
    sam2_config = config["model"]
    sam2 = SAM2Wrapper(
        checkpoint_path=sam2_config.get("checkpoint_path"),
        device=sam2_config.get("device", "cuda"),
    )
    
    # Get test dataset
    test_loader = data_loaders["test"]
    
    # Storage for predictions and ground truths
    all_pred_masks = []
    all_gt_masks = []
    all_pred_scores = []
    
    logger.info(f"Evaluating SAM2 on {len(test_loader)} test images")
    
    # Process each test image
    for idx in range(len(test_loader)):
        sample = test_loader[idx]
        image = sample["image_array"]
        
        # Encode image
        inference_state = sam2.set_image(image)
        
        # Get ground truth masks
        gt_masks = test_loader.get_instance_masks(idx)
        
        # Generate predictions for each ground truth object
        pred_masks = []
        pred_scores = []
        
        prompt_config = config.get("prompts", {})
        prompt_type = prompt_config.get("type", "point")
        
        for gt_mask in gt_masks:
            if prompt_type == "point":
                # Generate point prompts
                coords, labels = generate_point_prompt(
                    gt_mask,
                    num_points=prompt_config.get("num_points", 1),
                    strategy=prompt_config.get("strategy", "centroid"),
                )
                
                if len(coords) > 0:
                    result = sam2.predict_with_points(
                        inference_state,
                        coords,
                        labels,
                    )
                    pred_masks.append(result["masks"][0])
                    pred_scores.append(result["scores"][0])
            
            elif prompt_type == "box":
                # Generate box prompt
                box = generate_box_prompt(gt_mask)
                
                result = sam2.predict_with_box(
                    inference_state,
                    box,
                )
                pred_masks.append(result["masks"][0])
                pred_scores.append(result["scores"][0])
        
        all_pred_masks.extend(pred_masks)
        all_gt_masks.extend(gt_masks)
        all_pred_scores.extend(pred_scores)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_loader)} images")
    
    logger.info(f"SAM2 inference complete: {len(all_pred_masks)} predictions")
    
    # Compute geometric metrics
    metrics = aggregate_geometric_metrics(
        all_pred_masks,
        all_gt_masks,
        all_pred_scores,
    )
    
    # Log metrics
    tracker.log_metrics_dict(metrics)
    
    logger.info("SAM2 metrics computed:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return {
        "model": "SAM2",
        "pred_masks": all_pred_masks,
        "gt_masks": all_gt_masks,
        "pred_scores": all_pred_scores,
        "metrics": metrics,
    }


def run_sam3_experiment(
    config: Dict,
    data_loaders: Dict,
    tracker: ExperimentTracker,
) -> Dict:
    """
    Run SAM3 experiment with concept-driven segmentation.
    
    This function evaluates SAM3 using text prompts and measures
    performance using both geometric and semantic metrics.
    
    Args:
        config: Experiment configuration
        data_loaders: Dictionary of dataset loaders
        tracker: Experiment tracker for logging
        
    Returns:
        Dictionary with SAM3 results and metrics
    """
    logger.info("Starting SAM3 experiment")
    
    # Initialize SAM3 model
    sam3_config = config["model"]
    sam3 = SAM3Wrapper(
        checkpoint_path=sam3_config.get("checkpoint_path"),
        device=sam3_config.get("device", "cuda"),
    )
    
    # Create prompt templates
    prompt_template = create_mineapple_prompts()
    
    # Get test dataset
    test_loader = data_loaders["test"]
    
    # Storage for predictions and ground truths
    all_pred_masks = []
    all_gt_masks = []
    all_pred_labels = []
    all_gt_labels = []
    all_pred_attributes = []
    all_gt_attributes = []
    all_text_prompts = []
    
    logger.info(f"Evaluating SAM3 on {len(test_loader)} test images")
    
    # Get text prompts from config
    text_prompts_config = config.get("text_prompts", ["apples"])
    
    # Process each test image
    for idx in range(len(test_loader)):
        sample = test_loader[idx]
        image = sample["image"]
        
        # Encode image
        inference_state = sam3.set_image(image)
        
        # Get ground truth data
        gt_masks = test_loader.get_instance_masks(idx)
        gt_labels = test_loader.get_instance_labels(idx)
        gt_attributes = test_loader.get_instance_attributes(idx)
        
        # Run predictions for each text prompt
        for text_prompt in text_prompts_config:
            # Generate prompt from template
            if isinstance(text_prompt, dict):
                # Attribute-based prompt
                prompt_text = prompt_template.generate_attribute_prompt(
                    concept_name=text_prompt["concept"],
                    attributes=text_prompt.get("attributes", {}),
                )
            else:
                # Simple concept prompt
                prompt_text = text_prompt
            
            # Segment with text
            result = sam3.segment_with_text(
                inference_state,
                prompt_text,
            )
            
            # Store predictions
            for mask, label in zip(result["masks"], result["labels"]):
                all_pred_masks.append(mask)
                all_pred_labels.append(label)
                all_text_prompts.append(prompt_text)
                
                # Extract attributes from label if present
                # Simplified attribute extraction
                attributes = {
                    "ripeness": "unknown",
                    "color": "unknown",
                    "health": "unknown",
                }
                all_pred_attributes.append(attributes)
        
        # Store ground truths
        all_gt_masks.extend(gt_masks)
        all_gt_labels.extend(gt_labels)
        all_gt_attributes.extend(gt_attributes)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_loader)} images")
    
    logger.info(f"SAM3 inference complete: {len(all_pred_masks)} predictions")
    
    # Compute geometric metrics
    geometric_metrics = aggregate_geometric_metrics(
        all_pred_masks,
        all_gt_masks,
    )
    
    # Compute concept-level metrics
    concept_metrics = aggregate_concept_metrics(
        all_pred_masks,
        all_gt_masks,
        all_pred_labels,
        all_gt_labels,
        all_pred_attributes,
        all_gt_attributes,
        all_text_prompts,
    )
    
    # Combine metrics
    metrics = {**geometric_metrics, **concept_metrics}
    
    # Log metrics
    tracker.log_metrics_dict(metrics)
    
    logger.info("SAM3 metrics computed:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return {
        "model": "SAM3",
        "pred_masks": all_pred_masks,
        "gt_masks": all_gt_masks,
        "pred_labels": all_pred_labels,
        "gt_labels": all_gt_labels,
        "pred_attributes": all_pred_attributes,
        "gt_attributes": all_gt_attributes,
        "text_prompts": all_text_prompts,
        "metrics": metrics,
    }


def run_comparison_experiment(
    sam2_config_path: str,
    sam3_config_path: str,
    output_dir: str,
) -> Dict:
    """
    Run comparative analysis between SAM2 and SAM3.
    
    This orchestrates both experiments and generates comparative
    visualizations and statistical analyses highlighting the gap.
    
    Args:
        sam2_config_path: Path to SAM2 configuration
        sam3_config_path: Path to SAM3 configuration
        output_dir: Directory for comparison outputs
        
    Returns:
        Dictionary with comparative results
    """
    logger.info("Starting SAM2 vs SAM3 comparison experiment")
    
    # Load configurations
    sam2_config = load_config(sam2_config_path)
    sam3_config = load_config(sam3_config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment tracker
    tracker = ExperimentTracker(
        experiment_name="sam2_vs_sam3_comparison",
        output_dir=output_dir,
        config={
            "sam2_config": sam2_config,
            "sam3_config": sam3_config,
        }
    )
    
    # Create data loaders
    data_root = sam2_config["dataset"]["root"]
    data_loaders = create_data_loaders(data_root)
    
    # Run SAM2 experiment
    logger.info("=" * 60)
    logger.info("RUNNING SAM2 EXPERIMENT")
    logger.info("=" * 60)
    sam2_results = run_sam2_experiment(sam2_config, data_loaders, tracker)
    
    # Run SAM3 experiment
    logger.info("=" * 60)
    logger.info("RUNNING SAM3 EXPERIMENT")
    logger.info("=" * 60)
    sam3_results = run_sam3_experiment(sam3_config, data_loaders, tracker)
    
    # Generate comparative visualizations
    logger.info("Generating comparative visualizations")
    
    # Select example images for qualitative comparison
    test_loader = data_loaders["test"]
    num_examples = min(5, len(test_loader))
    
    for idx in range(num_examples):
        sample = test_loader[idx]
        image = sample["image_array"]
        
        # Get subset of masks for this image
        # Simplified: assume masks are in order
        # Production code would properly associate masks with images
        
        visualize_sam2_vs_sam3_comparison(
            image=image,
            sam2_masks=sam2_results["pred_masks"][idx*5:(idx+1)*5],
            sam3_masks=sam3_results["pred_masks"][idx*5:(idx+1)*5],
            sam2_prompts="Point/Box prompts",
            sam3_prompts="Text: " + ", ".join(sam3_results["text_prompts"][:3]),
            title=f"Comparison Example {idx+1}",
            save_path=str(tracker.visualizations_dir / f"comparison_{idx+1}.png"),
        )
    
    # Plot metric comparison
    common_metrics = [
        "mean_iou",
        "mean_boundary_f1",
        "mean_dice",
    ]
    
    plot_metric_comparison(
        metrics_sam2=sam2_results["metrics"],
        metrics_sam3=sam3_results["metrics"],
        metric_names=common_metrics,
        title="SAM2 vs SAM3: Geometric Performance",
        save_path=str(tracker.visualizations_dir / "metrics_comparison.png"),
    )
    
    # Plot IoU distributions
    sam2_ious = [sam2_results["metrics"]["mean_iou"]] * len(sam2_results["pred_masks"])
    sam3_ious = [sam3_results["metrics"]["mean_iou"]] * len(sam3_results["pred_masks"])
    
    plot_iou_distribution(
        sam2_ious=sam2_ious,
        sam3_ious=sam3_ious,
        title="IoU Distribution: SAM2 vs SAM3",
        save_path=str(tracker.visualizations_dir / "iou_distribution.png"),
    )
    
    # Compute and save comparison summary
    comparison_summary = {
        "sam2": {
            "num_predictions": len(sam2_results["pred_masks"]),
            "key_metrics": {
                "mean_iou": sam2_results["metrics"]["mean_iou"],
                "mean_boundary_f1": sam2_results["metrics"]["mean_boundary_f1"],
                "mean_dice": sam2_results["metrics"]["mean_dice"],
            }
        },
        "sam3": {
            "num_predictions": len(sam3_results["pred_masks"]),
            "key_metrics": {
                "mean_iou": sam3_results["metrics"]["mean_iou"],
                "mean_boundary_f1": sam3_results["metrics"]["mean_boundary_f1"],
                "mean_dice": sam3_results["metrics"]["mean_dice"],
            }
        },
        "gap_analysis": {
            "iou_diff": sam3_results["metrics"]["mean_iou"] - sam2_results["metrics"]["mean_iou"],
            "boundary_f1_diff": sam3_results["metrics"]["mean_boundary_f1"] - sam2_results["metrics"]["mean_boundary_f1"],
        }
    }
    
    tracker.save_results(comparison_summary, "comparison_summary.json")
    
    # Finalize experiment
    tracker.finalize()
    
    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info(f"SAM2 mean IoU: {sam2_results['metrics']['mean_iou']:.4f}")
    logger.info(f"SAM3 mean IoU: {sam3_results['metrics']['mean_iou']:.4f}")
    logger.info(f"Gap: {comparison_summary['gap_analysis']['iou_diff']:.4f}")
    logger.info(f"Results saved to: {tracker.experiment_dir}")
    logger.info("=" * 60)
    
    return comparison_summary


def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run SAM2/SAM3 experiments on MineApple dataset"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sam2", "sam3", "compare"],
        default="compare",
        help="Experiment mode: sam2 only, sam3 only, or comparison"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config file (for sam2 or sam3 mode)"
    )
    
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2_mineapple.yml",
        help="Path to SAM2 config (for compare mode)"
    )
    
    parser.add_argument(
        "--sam3-config",
        type=str,
        default="configs/sam3_mineapple.yml",
        help="Path to SAM3 config (for compare mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_dir=f"{args.output_dir}/logs",
        experiment_name=args.mode,
        log_level=args.log_level,
    )
    
    logger.info("Starting experiment runner")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.mode == "compare":
            # Run comparison experiment
            results = run_comparison_experiment(
                sam2_config_path=args.sam2_config,
                sam3_config_path=args.sam3_config,
                output_dir=args.output_dir,
            )
        
        elif args.mode in ["sam2", "sam3"]:
            if not args.config:
                raise ValueError(f"--config required for {args.mode} mode")
            
            config = load_config(args.config)
            
            # Create data loaders
            data_root = config["dataset"]["root"]
            data_loaders = create_data_loaders(data_root)
            
            # Create tracker
            tracker = ExperimentTracker(
                experiment_name=args.mode,
                output_dir=args.output_dir,
                config=config,
            )
            
            # Run appropriate experiment
            if args.mode == "sam2":
                results = run_sam2_experiment(config, data_loaders, tracker)
            else:
                results = run_sam3_experiment(config, data_loaders, tracker)
            
            # Save results
            tracker.save_results(results["metrics"])
            tracker.finalize()
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
