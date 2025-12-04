#!/usr/bin/env python3
"""
Logging and Experiment Tracking Utilities

This module provides comprehensive logging infrastructure for experiments,
including structured logging, metric tracking, experiment versioning, and
results persistence. Ensures reproducibility and facilitates analysis.

All experiments are logged with configurations, random seeds, timestamps,
and complete metric histories for thorough documentation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: str,
    experiment_name: str,
    log_level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure logging infrastructure for experiment.
    
    Sets up both file and console logging with appropriate formatting
    for easy debugging and experiment tracking.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of current experiment
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"{experiment_name}_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with simplified output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            fmt='[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for experiment '{experiment_name}'")
    logger.info(f"Log file: {log_file}")
    
    return root_logger


class ExperimentTracker:
    """
    Comprehensive experiment tracking and results management.
    
    This class handles experiment configuration logging, metric tracking,
    results persistence, and experiment comparison utilities.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        config: Optional[Dict] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Unique identifier for this experiment
            output_dir: Directory to save all experiment outputs
            config: Experiment configuration dictionary
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create experiment directory structure
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = self.experiment_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Initialize tracking data structures
        self.metrics_history: Dict[str, List] = {}
        self.start_time = datetime.now()
        
        # Save experiment configuration
        self._save_config()
        
        logger.info(f"Experiment tracker initialized: {experiment_name}")
    
    def _save_config(self):
        """Save experiment configuration to YAML file."""
        config_file = self.experiment_dir / "config.yml"
        
        config_data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "config": self.config,
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.debug(f"Saved configuration to {config_file}")
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
    ):
        """
        Log a metric value with optional step counter.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional iteration or epoch number
        """
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        entry = {
            "value": float(value),
            "timestamp": datetime.now().isoformat(),
        }
        
        if step is not None:
            entry["step"] = step
        
        self.metrics_history[metric_name].append(entry)
        
        logger.debug(f"Logged metric {metric_name} = {value:.4f}")
    
    def log_metrics_dict(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step counter
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
        
        logger.info(f"Logged {len(metrics)} metrics at step {step}")
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = "results.json",
    ):
        """
        Save experiment results to JSON file.
        
        Args:
            results: Dictionary of results to save
            filename: Output filename
        """
        results_file = self.experiment_dir / filename
        
        # Add metadata
        results_with_meta = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "results": results,
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
    
    def save_metrics_history(self):
        """Save complete metrics history to JSON file."""
        metrics_file = self.metrics_dir / "metrics_history.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Saved metrics history to {metrics_file}")
    
    def get_metric_summary(
        self,
        metric_name: str,
    ) -> Dict[str, float]:
        """
        Compute summary statistics for a metric.
        
        Args:
            metric_name: Name of metric to summarize
            
        Returns:
            Dictionary with mean, std, min, max, final values
        """
        if metric_name not in self.metrics_history:
            logger.warning(f"Metric '{metric_name}' not found")
            return {}
        
        values = [entry["value"] for entry in self.metrics_history[metric_name]]
        
        import numpy as np
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "final": float(values[-1]) if values else 0.0,
            "count": len(values),
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their summaries
        """
        summaries = {}
        
        for metric_name in self.metrics_history.keys():
            summaries[metric_name] = self.get_metric_summary(metric_name)
        
        return summaries
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict,
        checkpoint_name: str = "checkpoint",
    ):
        """
        Save experiment checkpoint for resumption.
        
        Args:
            checkpoint_data: Data to checkpoint
            checkpoint_name: Name for checkpoint file
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_name}.json"
        
        checkpoint_with_meta = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "data": checkpoint_data,
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_with_meta, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    
    def load_checkpoint(
        self,
        checkpoint_name: str = "checkpoint",
    ) -> Optional[Dict]:
        """
        Load experiment checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Checkpoint data dictionary or None if not found
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_file}")
            return None
        
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_file}")
        return checkpoint.get("data")
    
    def finalize(self):
        """
        Finalize experiment and save all tracking data.
        
        Call this at the end of an experiment to ensure everything is saved.
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Save final metrics history
        self.save_metrics_history()
        
        # Save experiment summary
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "duration_human": str(end_time - self.start_time),
            "metrics_summary": self.get_all_metrics_summary(),
        }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment finalized. Duration: {duration:.2f}s")
        logger.info(f"Results saved to: {self.experiment_dir}")


def load_experiment_results(
    experiment_dir: str,
) -> Dict[str, Any]:
    """
    Load complete results from a previous experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with all experiment data
    """
    exp_dir = Path(experiment_dir)
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    results = {}
    
    # Load config
    config_file = exp_dir / "config.yml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            results["config"] = yaml.safe_load(f)
    
    # Load summary
    summary_file = exp_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results["summary"] = json.load(f)
    
    # Load metrics history
    metrics_file = exp_dir / "metrics" / "metrics_history.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results["metrics_history"] = json.load(f)
    
    logger.info(f"Loaded experiment results from {experiment_dir}")
    return results


def compare_experiments(
    experiment_dirs: List[str],
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        metric_names: List of metric names to compare
        
    Returns:
        Dictionary mapping experiment names to their metric values
    """
    comparison = {}
    
    for exp_dir in experiment_dirs:
        exp_results = load_experiment_results(exp_dir)
        exp_name = exp_results["config"]["experiment_name"]
        
        comparison[exp_name] = {}
        metrics_summary = exp_results["summary"]["metrics_summary"]
        
        for metric_name in metric_names:
            if metric_name in metrics_summary:
                comparison[exp_name][metric_name] = metrics_summary[metric_name]["mean"]
    
    logger.info(f"Compared {len(experiment_dirs)} experiments on {len(metric_names)} metrics")
    return comparison
