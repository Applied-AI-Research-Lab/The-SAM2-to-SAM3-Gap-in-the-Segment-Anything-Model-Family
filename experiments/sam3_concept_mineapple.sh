#!/bin/bash
# SAM3 Concept-Driven Experiment on MineApple Dataset
#
# This script runs the complete SAM3 evaluation pipeline using concept-driven
# segmentation with text prompts. It demonstrates semantic understanding and
# attribute-based reasoning capabilities beyond SAM2's geometric approach.
#
# Usage:
#   bash experiments/sam3_concept_mineapple.sh

set -e  # Exit on error

echo "=========================================="
echo "SAM3 Concept-Driven Experiment on MineApple"
echo "=========================================="

# Configuration paths
CONFIG_PATH="configs/sam3_mineapple.yml"
OUTPUT_DIR="results/sam3_concept"
DATA_ROOT="data/mineapple"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Download and prepare MineApple dataset (if not already present)
echo ""
echo "[1/5] Checking MineApple dataset..."
if [ ! -d "$DATA_ROOT/images" ]; then
    echo "Dataset not found. Downloading MineApple dataset..."
    python data/download_mineapple.py --output-dir "$DATA_ROOT"
else
    echo "Dataset already present at $DATA_ROOT"
fi

# Step 2: Verify SAM3 installation and authenticate with HuggingFace
echo ""
echo "[2/5] Verifying SAM3 installation..."
python -c "import sam3; print(f'SAM3 installed successfully')" || {
    echo "ERROR: SAM3 not installed. Installing from GitHub..."
    pip install git+https://github.com/facebookresearch/sam3.git
}

# Check HuggingFace authentication for SAM3 model access
echo "Checking HuggingFace authentication..."
python -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); assert token is not None, 'No HF token found'" || {
    echo "WARNING: HuggingFace authentication not found."
    echo "SAM3 requires HF authentication. Please run:"
    echo "  huggingface-cli login"
    echo "Then rerun this script."
    exit 1
}

# Step 3: Download SAM3 checkpoint if not using HuggingFace
CHECKPOINT_PATH="checkpoints/sam3_model.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "SAM3 checkpoint not found locally. Will use HuggingFace model hub."
fi

# Step 4: Run SAM3 experiment with text prompts
echo ""
echo "[3/5] Running SAM3 concept segmentation experiment..."
echo "Configuration: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Using text prompts:"
echo "  - 'apples'"
echo "  - 'ripe apples'"
echo "  - 'unripe apples'"
echo "  - 'red apples'"
echo "  - 'green apples'"

python src/run_experiment.py \
    --mode sam3 \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --log-level INFO

# Step 5: Generate summary report with concept-level metrics
echo ""
echo "[4/5] Generating experiment summary..."

# Extract key metrics from results
if [ -f "$OUTPUT_DIR/sam3/summary.json" ]; then
    echo ""
    echo "=========================================="
    echo "SAM3 EXPERIMENT RESULTS"
    echo "=========================================="
    
    # Use Python to parse and display JSON results
    python << EOF
import json
with open("$OUTPUT_DIR/sam3/summary.json") as f:
    results = json.load(f)
    
print(f"\nExperiment: {results['experiment_name']}")
print(f"Duration: {results['duration_human']}")

print("\nGeometric Metrics:")
geometric_metrics = ['mean_iou', 'mean_boundary_f1', 'mean_dice']
for metric_name in geometric_metrics:
    if metric_name in results['metrics_summary']:
        metric_data = results['metrics_summary'][metric_name]
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            print(f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")

print("\nConcept-Level Metrics:")
concept_metrics = ['mean_concept_recall', 'mean_concept_precision', 'semantic_grounding_accuracy']
for metric_name in concept_metrics:
    if metric_name in results['metrics_summary']:
        metric_data = results['metrics_summary'][metric_name]
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            print(f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")

print("\nAttribute Understanding:")
attribute_metrics = ['ripeness', 'color', 'health']
for attr in attribute_metrics:
    metric_name = f'{attr}_accuracy'  
    if metric_name in results['metrics_summary']:
        metric_data = results['metrics_summary'][metric_name]
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            print(f"  {attr}: {metric_data['mean']:.4f}")

EOF

    echo ""
    echo "Full results saved to: $OUTPUT_DIR/sam3/"
    echo "Visualizations saved to: $OUTPUT_DIR/sam3/visualizations/"
else
    echo "ERROR: Results file not found. Experiment may have failed."
    exit 1
fi

echo ""
echo "[5/5] Experiment complete!"
echo ""
echo "=========================================="
echo "SAM3 Concept-Driven Experiment Complete!"
echo "=========================================="
