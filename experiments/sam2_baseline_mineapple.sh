#!/bin/bash
# SAM2 Baseline Experiment on MineApple Dataset
#
# This script runs the complete SAM2 evaluation pipeline using prompt-based
# segmentation with geometric prompts (points and boxes). It establishes
# the baseline performance for comparison with SAM3.
#
# Usage:
#   bash experiments/sam2_baseline_mineapple.sh

set -e  # Exit on error

echo "=========================================="
echo "SAM2 Baseline Experiment on MineApple"
echo "=========================================="

# Configuration paths
CONFIG_PATH="configs/sam2_mineapple.yml"
OUTPUT_DIR="results/sam2_baseline"
DATA_ROOT="data/mineapple"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Download and prepare MineApple dataset (if not already present)
echo ""
echo "[1/4] Checking MineApple dataset..."
if [ ! -d "$DATA_ROOT/images" ]; then
    echo "Dataset not found. Downloading MineApple dataset..."
    python data/download_mineapple.py --output-dir "$DATA_ROOT"
else
    echo "Dataset already present at $DATA_ROOT"
fi

# Step 2: Verify SAM2 installation and download checkpoints
echo ""
echo "[2/4] Verifying SAM2 installation..."
python -c "import sam2; print(f'SAM2 version: {sam2.__version__}')" || {
    echo "ERROR: SAM2 not installed. Please run: pip install -r requirements.txt"
    exit 1
}

# Download SAM2 checkpoint if not present
CHECKPOINT_PATH="checkpoints/sam2_hiera_large.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading SAM2 checkpoint..."
    mkdir -p checkpoints
    # Note: Update URL with actual SAM2 checkpoint location
    wget -O "$CHECKPOINT_PATH" "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" || {
        echo "WARNING: Checkpoint download failed. Will use HuggingFace fallback."
    }
fi

# Step 3: Run SAM2 experiment with geometric prompts
echo ""
echo "[3/4] Running SAM2 segmentation experiment..."
echo "Configuration: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"

python src/run_experiment.py \
    --mode sam2 \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --log-level INFO

# Step 4: Generate summary report
echo ""
echo "[4/4] Generating experiment summary..."

# Extract key metrics from results
if [ -f "$OUTPUT_DIR/sam2/summary.json" ]; then
    echo ""
    echo "=========================================="
    echo "SAM2 EXPERIMENT RESULTS"
    echo "=========================================="
    
    # Use Python to parse and display JSON results
    python << EOF
import json
with open("$OUTPUT_DIR/sam2/summary.json") as f:
    results = json.load(f)
    
print(f"\nExperiment: {results['experiment_name']}")
print(f"Duration: {results['duration_human']}")
print("\nKey Metrics:")
for metric_name, metric_data in sorted(results['metrics_summary'].items()):
    if isinstance(metric_data, dict) and 'mean' in metric_data:
        print(f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")

EOF

    echo ""
    echo "Full results saved to: $OUTPUT_DIR/sam2/"
    echo "Visualizations saved to: $OUTPUT_DIR/sam2/visualizations/"
else
    echo "ERROR: Results file not found. Experiment may have failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "SAM2 Baseline Experiment Complete!"
echo "=========================================="
