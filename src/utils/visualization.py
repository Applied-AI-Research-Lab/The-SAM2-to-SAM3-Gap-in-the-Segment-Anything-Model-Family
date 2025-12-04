#!/usr/bin/env python3
"""
Visualization Utilities for SAM2 and SAM3

This module provides plotting and visualization functions for comparing
segmentation results, displaying mask overlays, creating metric dashboards,
and generating publication-quality figures for the paper.

Visualizations help understand the SAM2-to-SAM3 gap through qualitative
comparisons and quantitative metric plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image

logger = logging.getLogger(__name__)

# Set publication-quality plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize a single segmentation mask overlaid on the image.
    
    Args:
        image: RGB image array (H, W, 3)
        mask: Binary mask array (H, W)
        color: RGB color tuple for mask overlay
        alpha: Transparency of mask overlay
        title: Optional title for the plot
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display image
    ax.imshow(image)
    
    # Create colored mask overlay
    overlay = np.zeros_like(image)
    overlay[mask] = color
    
    # Blend with image
    ax.imshow(overlay, alpha=alpha)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.debug(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_multiple_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize multiple instance masks with distinct colors.
    
    Args:
        image: RGB image array
        masks: List of binary masks
        labels: Optional list of instance labels
        title: Optional plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    ax.imshow(image)
    
    # Generate distinct colors for each instance
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    # Overlay each mask
    for i, mask in enumerate(masks):
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = colors[i]
        ax.imshow(overlay, alpha=0.5)
        
        # Add label if provided
        if labels:
            # Find centroid for label placement
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                ax.text(cx, cy, labels[i], color='white', 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=colors[i][:3], alpha=0.7))
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.debug(f"Saved multi-mask visualization to {save_path}")
    
    plt.close()


def visualize_sam2_vs_sam3_comparison(
    image: np.ndarray,
    sam2_masks: List[np.ndarray],
    sam3_masks: List[np.ndarray],
    sam2_prompts: Optional[str] = None,
    sam3_prompts: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Create side-by-side comparison of SAM2 and SAM3 segmentations.
    
    This visualization highlights the key differences between prompt-based
    and concept-driven segmentation approaches.
    
    Args:
        image: Original RGB image
        sam2_masks: SAM2 predicted masks
        sam3_masks: SAM3 predicted masks
        sam2_prompts: Description of SAM2 prompts used
        sam3_prompts: Description of SAM3 text prompts used
        title: Overall comparison title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # SAM2 results
    axes[1].imshow(image)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(sam2_masks)))
    for i, mask in enumerate(sam2_masks):
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = colors[i]
        axes[1].imshow(overlay, alpha=0.6)
    
    axes[1].set_title(f"SAM2 (Prompt-Based)\n{len(sam2_masks)} instances", 
                     fontsize=12, fontweight='bold')
    if sam2_prompts:
        axes[1].text(0.5, -0.1, f"Prompts: {sam2_prompts}", 
                    transform=axes[1].transAxes, ha='center', fontsize=9)
    axes[1].axis('off')
    
    # SAM3 results
    axes[2].imshow(image)
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(sam3_masks)))
    for i, mask in enumerate(sam3_masks):
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = colors[i]
        axes[2].imshow(overlay, alpha=0.6)
    
    axes[2].set_title(f"SAM3 (Concept-Driven)\n{len(sam3_masks)} instances", 
                     fontsize=12, fontweight='bold')
    if sam3_prompts:
        axes[2].text(0.5, -0.1, f"Concepts: {sam3_prompts}", 
                    transform=axes[2].transAxes, ha='center', fontsize=9)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved comparison visualization to {save_path}")
    
    plt.close()


def plot_metric_comparison(
    metrics_sam2: Dict[str, float],
    metrics_sam3: Dict[str, float],
    metric_names: List[str],
    title: str = "SAM2 vs SAM3 Performance",
    save_path: Optional[str] = None,
):
    """
    Create bar chart comparing metrics between SAM2 and SAM3.
    
    Args:
        metrics_sam2: Dictionary of SAM2 metrics
        metrics_sam3: Dictionary of SAM3 metrics
        metric_names: List of metric keys to compare
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    sam2_values = [metrics_sam2.get(m, 0) for m in metric_names]
    sam3_values = [metrics_sam3.get(m, 0) for m in metric_names]
    
    bars1 = ax.bar(x - width/2, sam2_values, width, label='SAM2', color='#4472C4')
    bars2 = ax.bar(x + width/2, sam3_values, width, label='SAM3', color='#ED7D31')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved metric comparison to {save_path}")
    
    plt.close()


def plot_iou_distribution(
    sam2_ious: List[float],
    sam3_ious: List[float],
    title: str = "IoU Distribution Comparison",
    save_path: Optional[str] = None,
):
    """
    Plot histogram comparing IoU distributions between SAM2 and SAM3.
    
    Args:
        sam2_ious: List of SAM2 IoU scores
        sam3_ious: List of SAM3 IoU scores
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    
    ax.hist(sam2_ious, bins=bins, alpha=0.6, label='SAM2', color='#4472C4')
    ax.hist(sam3_ious, bins=bins, alpha=0.6, label='SAM3', color='#ED7D31')
    
    ax.axvline(np.mean(sam2_ious), color='#4472C4', linestyle='--', 
              label=f'SAM2 mean: {np.mean(sam2_ious):.3f}')
    ax.axvline(np.mean(sam3_ious), color='#ED7D31', linestyle='--', 
              label=f'SAM3 mean: {np.mean(sam3_ious):.3f}')
    
    ax.set_xlabel('IoU Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved IoU distribution to {save_path}")
    
    plt.close()


def plot_attribute_confusion_matrix(
    confusion_matrix: np.ndarray,
    attribute_values: List[str],
    title: str = "Attribute Confusion Matrix",
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix for attribute classification.
    
    Args:
        confusion_matrix: 2D array of confusion counts
        attribute_values: List of attribute value names
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=attribute_values, yticklabels=attribute_values,
               ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Ground Truth', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_size_stratified_performance(
    size_categories: List[str],
    sam2_scores: List[float],
    sam3_scores: List[float],
    title: str = "Performance by Object Size",
    save_path: Optional[str] = None,
):
    """
    Plot performance metrics stratified by object size.
    
    Args:
        size_categories: List of size category names
        sam2_scores: SAM2 scores for each category
        sam3_scores: SAM3 scores for each category
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(size_categories))
    width = 0.35
    
    ax.bar(x - width/2, sam2_scores, width, label='SAM2', color='#4472C4')
    ax.bar(x + width/2, sam3_scores, width, label='SAM3', color='#ED7D31')
    
    ax.set_ylabel('Mean IoU', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved size-stratified plot to {save_path}")
    
    plt.close()


def create_qualitative_grid(
    images: List[np.ndarray],
    sam2_masks_list: List[List[np.ndarray]],
    sam3_masks_list: List[List[np.ndarray]],
    titles: List[str],
    save_path: Optional[str] = None,
):
    """
    Create grid of qualitative examples comparing SAM2 and SAM3.
    
    Args:
        images: List of original images
        sam2_masks_list: List of SAM2 mask sets
        sam3_masks_list: List of SAM3 mask sets
        titles: List of titles for each example
        save_path: Optional save path
    """
    n_examples = len(images)
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 5 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        # Original
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(titles[i] if i < len(titles) else f"Example {i+1}")
        axes[i, 0].axis('off')
        
        # SAM2
        axes[i, 1].imshow(images[i])
        for mask in sam2_masks_list[i]:
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [0, 0, 1, 0.5]
            axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"SAM2: {len(sam2_masks_list[i])} masks")
        axes[i, 1].axis('off')
        
        # SAM3
        axes[i, 2].imshow(images[i])
        for mask in sam3_masks_list[i]:
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [1, 0, 0, 0.5]
            axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"SAM3: {len(sam3_masks_list[i])} masks")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved qualitative grid to {save_path}")
    
    plt.close()


def plot_prompt_efficiency_curve(
    num_clicks: List[int],
    ious: List[float],
    title: str = "Interactive Refinement Efficiency",
    save_path: Optional[str] = None,
):
    """
    Plot IoU improvement curve as function of interactive clicks.
    
    Args:
        num_clicks: List of click counts
        ious: Corresponding IoU scores
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(num_clicks, ious, marker='o', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Interactive Clicks', fontweight='bold')
    ax.set_ylabel('Mean IoU', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Mark 0.9 IoU threshold
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='0.9 IoU target')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved efficiency curve to {save_path}")
    
    plt.close()


def create_paper_figure(
    figure_data: Dict,
    figure_type: str,
    save_path: str,
):
    """
    Create publication-quality figure for paper submission.
    
    This is a high-level function that generates complete figures
    with proper formatting, labels, and layout for academic publication.
    
    Args:
        figure_data: Dictionary with all required data for the figure
        figure_type: Type of figure (comparison, metrics, qualitative, etc.)
        save_path: Path to save the figure
    """
    if figure_type == "main_comparison":
        # Create main comparison figure showing SAM2 vs SAM3 gap
        visualize_sam2_vs_sam3_comparison(
            image=figure_data["image"],
            sam2_masks=figure_data["sam2_masks"],
            sam3_masks=figure_data["sam3_masks"],
            sam2_prompts=figure_data.get("sam2_prompts"),
            sam3_prompts=figure_data.get("sam3_prompts"),
            title=figure_data.get("title"),
            save_path=save_path,
        )
    
    elif figure_type == "performance_metrics":
        plot_metric_comparison(
            metrics_sam2=figure_data["sam2_metrics"],
            metrics_sam3=figure_data["sam3_metrics"],
            metric_names=figure_data["metric_names"],
            title=figure_data.get("title", "Performance Comparison"),
            save_path=save_path,
        )
    
    else:
        logger.warning(f"Unknown figure type: {figure_type}")
    
    logger.info(f"Created paper figure: {save_path}")
