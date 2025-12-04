# Experimental Methodology

## Overview

This document details the experimental design, evaluation protocols, and methodological decisions for comparing SAM2 and SAM3 on the MineApple orchard dataset. We establish rigorous procedures to ensure reproducible, fair, and meaningful comparisons that illuminate the architectural gap between these models.

## Dataset: MineApple Orchard Imagery

### Data Source
- **Origin**: University of Minnesota Digital Conservancy
- **URL**: https://conservancy.umn.edu/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/download
- **Size**: Approximately 2.5 GB compressed
- **Images**: ~1200 orchard photographs
- **Domain**: Apple orchards under natural outdoor conditions

### Annotation Schema
Each image contains instance-level annotations with:

#### Spatial Information
- **Segmentation masks**: Pixel-precise binary masks
- **Bounding boxes**: Tight rectangular bounds in [x, y, w, h] format
- **Instance IDs**: Unique identifiers for tracking

#### Semantic Attributes
- **Ripeness**: {ripe, unripe, overripe, intermediate}
- **Color**: {red, green, yellow, mixed}
- **Health**: {healthy, damaged, diseased, defective}
- **Size**: {small, medium, large} (normalized by image resolution)

### Dataset Splits
We employ stratified random splitting to ensure balanced distributions:

- **Training**: 70% (~840 images)
- **Validation**: 15% (~180 images)
- **Test**: 15% (~180 images)

**Stratification Criteria**:
- Apple count per image
- Attribute distribution balance
- Lighting condition diversity

**Random Seed**: 42 (for reproducibility)

## Model Configurations

### SAM2 Setup

#### Architecture
- **Model Variant**: SAM2-Hiera-Large
- **Parameters**: 224.4M
- **Image Encoder**: Hierarchical Vision Transformer
- **Memory Module**: Enabled for temporal consistency
- **Checkpoint**: Official Facebook Research weights

#### Prompt Configuration
We evaluate three prompting strategies:

1. **Point Prompts**
   - Strategy: Centroid extraction from ground truth masks
   - Num points: 1 (primary), 3 (ablation)
   - Label: Positive (foreground) only

2. **Box Prompts**
   - Extraction: Tight bounding box from mask
   - Padding: 0 pixels (exact bounds)
   - Jitter: None (ground truth quality)

3. **Hybrid Prompts** (ablation)
   - Combination of point + box
   - Tests prompt complementarity

**Rationale**: We use ground truth-derived prompts to establish SAM2's upper-bound performance, isolating the architectural gap from prompt quality effects.

### SAM3 Setup

#### Architecture
- **Model**: SAM3 Full Model
- **Parameters**: 848M total
  - Vision Encoder: 450M
  - Text Encoder: 300M
  - Fusion + Decoder: 98M
- **Resolution**: 1008×1008 pixels
- **Checkpoint**: HuggingFace model hub (official release)

#### Text Prompt Configuration
We design a hierarchical prompt set:

1. **Base Concepts** (general)
   - "apples"
   - "apple fruit"
   - "fruit instances"

2. **Attribute-Based Prompts** (specific)
   - Ripeness: "ripe apples", "unripe apples"
   - Color: "red apples", "green apples", "yellow apples"
   - Health: "healthy apples", "damaged apples"

3. **Compositional Prompts** (complex)
   - "ripe red apples"
   - "healthy green apples"
   - "damaged fruit"

4. **Negative Prompts** (contrastive)
   - "apples but not leaves"
   - "fruit excluding background"

**Rationale**: This hierarchy tests SAM3's semantic understanding at multiple levels of specificity and complexity.

## Evaluation Metrics

### Geometric Metrics (Both Models)

#### Intersection over Union (IoU)
$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

Where P = predicted mask, G = ground truth mask

**Thresholds**: 0.5, 0.75, 0.9, 0.95  
**Aggregation**: Mean IoU (mIoU) across all instances

#### Boundary F1 Score
$$\text{Boundary-F1} = \frac{2 \cdot \text{Precision}_{\text{boundary}} \cdot \text{Recall}_{\text{boundary}}}{\text{Precision}_{\text{boundary}} + \text{Recall}_{\text{boundary}}}$$

**Tolerance**: 2 pixels distance  
**Rationale**: Boundary precision is critical for robotic manipulation

#### Dice Coefficient
$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

**Usage**: Emphasizes overlap, sensitive to small objects

### Instance-Level Metrics

#### Precision and Recall
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

**Matching Criterion**: IoU ≥ 0.5

#### F1 Score
$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Semantic Metrics (SAM3 Only)

#### Concept Recall
$$\text{Concept Recall}_c = \frac{\text{Detected instances of concept } c}{\text{Total instances of concept } c}$$

**Per-concept and mean across all concepts**

#### Semantic Grounding Accuracy
$$\text{Grounding Acc} = \frac{\text{Correct text-to-mask alignments}}{\text{Total predictions}}$$

**Evaluation**: Text prompt semantically matches ground truth label

#### Attribute Understanding Accuracy
$$\text{Attr Acc}_{attr} = \frac{\text{Correct attribute predictions}}{\text{Total matched instances}}$$

**Attributes**: Ripeness, color, health (evaluated separately)

### Open-Vocabulary Metrics (SAM3 Only)

#### Seen vs. Unseen Performance
- **Seen concepts**: Present in training vocabulary
- **Unseen concepts**: Novel compositions or synonyms
- **Metric**: F1 score gap between seen and unseen

**Generalization Gap**:
$$\Delta_{\text{gen}} = F1_{\text{seen}} - F1_{\text{unseen}}$$

### Efficiency Metrics

#### Prompt Efficiency (SAM2)
- Clicks required to reach IoU ≥ 0.9
- Interactive refinement curve

#### Inference Time
- Per-image processing time (milliseconds)
- GPU: NVIDIA A100 40GB
- Batch size: 1 (fair comparison)

## Experimental Procedures

### Experiment 1: Baseline Geometric Performance

**Objective**: Establish both models' spatial accuracy

**Protocol**:
1. Load test set images
2. For SAM2: Generate geometric prompts from ground truth
3. For SAM3: Use base concept text prompts
4. Compute geometric metrics (IoU, Boundary F1, Dice)
5. Perform size-stratified analysis (small/medium/large objects)

**Expected Outcome**: Quantify geometric performance parity/gap

### Experiment 2: Semantic Understanding (SAM3)

**Objective**: Evaluate concept-level reasoning

**Protocol**:
1. Use attribute-based text prompts
2. Match predictions to ground truth by spatial IoU + semantic label
3. Compute concept recall and precision per attribute
4. Generate confusion matrices for attribute classification

**Expected Outcome**: Demonstrate SAM3's semantic capabilities unavailable to SAM2

### Experiment 3: Compositional Reasoning (SAM3)

**Objective**: Test multi-attribute understanding

**Protocol**:
1. Use compositional prompts (e.g., "ripe red apples")
2. Require predictions to match ALL specified attributes
3. Measure accuracy as function of attribute count

**Expected Outcome**: Show SAM3's compositional generalization

### Experiment 4: Open-Vocabulary Evaluation (SAM3)

**Objective**: Assess zero-shot generalization

**Protocol**:
1. Define seen concept set (common terms)
2. Generate unseen synonyms and paraphrases
3. Compare performance across seen/unseen splits

**Expected Outcome**: Quantify open-vocabulary capability

### Experiment 5: Interactive Efficiency (SAM2)

**Objective**: Measure prompt efficiency

**Protocol**:
1. Start with 1-point prompt
2. Iteratively add corrective clicks
3. Track IoU improvement curve
4. Measure clicks to convergence (IoU ≥ 0.9)

**Expected Outcome**: Characterize interactive refinement efficiency

### Experiment 6: Qualitative Comparison

**Objective**: Visual analysis of segmentation quality

**Protocol**:
1. Select diverse test images (varied conditions)
2. Generate side-by-side comparisons
3. Annotate with prompt types and predictions
4. Expert review for qualitative assessment

**Expected Outcome**: Illustrative examples of gap characteristics

## Statistical Analysis

### Significance Testing
- **Test**: Paired t-test on per-image IoU scores
- **Null Hypothesis**: No difference between SAM2 and SAM3 performance
- **Significance Level**: α = 0.05
- **Correction**: Bonferroni for multiple comparisons

### Effect Size
- **Cohen's d**: Standardized mean difference
- **Interpretation**:
  - Small: d = 0.2
  - Medium: d = 0.5
  - Large: d = 0.8

### Confidence Intervals
- **Bootstrap**: 10,000 resamples
- **Confidence Level**: 95%
- **Reported**: Mean ± 95% CI for all metrics

## Reproducibility Measures

### Environment Specification
```yaml
Python: 3.10+
PyTorch: 2.5.1
CUDA: 12.1
GPU: NVIDIA A100 40GB
OS: Ubuntu 22.04 LTS
```

### Seeds and Determinism
- Dataset split seed: 42
- NumPy random seed: 42
- PyTorch manual seed: 42
- CUDA deterministic mode: Enabled

### Version Control
- All code versioned in Git
- Experiment configurations saved with results
- Model checkpoints hashed and logged
- Data preprocessing pipeline frozen

### Data Management
- Raw images preserved unchanged
- Annotations validated against schema
- Train/val/test splits serialized
- Metadata logged for each experiment run

## Validation and Quality Control

### Annotation Quality
- Inter-annotator agreement: Kappa > 0.8
- Systematic review of boundary accuracy
- Attribute label consistency checks

### Model Validation
- Checkpoint integrity verification
- Gradient checks for training stability
- Output distribution analysis

### Experiment Validation
- Pilot runs on validation set
- Hyperparameter sensitivity analysis
- Ablation studies for key decisions

## Ethical Considerations

### Dataset Usage
- Public dataset with appropriate license
- University of Minnesota attribution
- No personally identifiable information

### Computational Resources
- Energy consumption logged
- Carbon footprint estimated
- Resource usage optimized

### Transparency
- All methodological decisions documented
- Negative results reported
- Limitations explicitly stated

## Limitations and Assumptions

### Assumptions
1. Ground truth annotations are accurate
2. Test set represents real orchard conditions
3. Prompt quality equals annotation quality for SAM2
4. Text prompts adequately express concepts for SAM3

### Limitations
1. Single crop type (apples) - generalization unclear
2. Limited attribute vocabulary
3. Controlled prompt generation - not interactive user study
4. Computational resources constrain model scales tested

### Threats to Validity

#### Internal Validity
- Prompt quality differences between models
- Mitigation: Use ground truth-derived prompts for SAM2

#### External Validity
- Dataset domain specificity
- Mitigation: Acknowledge scope, suggest multi-crop validation

#### Construct Validity
- Metrics may not capture all relevant performance aspects
- Mitigation: Use diverse metric suite covering geometric and semantic dimensions

## Fine-tuning Methodology (NEW!)

### Overview

In addition to inference-based evaluation, this repository provides complete fine-tuning infrastructure for adapting both SAM2 and SAM3 to custom datasets. This section documents the training methodology and best practices.

### Training Data Requirements

#### SAM2 Fine-tuning
**Format**: MineApple-style annotations
- Instance-level segmentation masks (polygon or RLE format)
- Bounding boxes for each instance
- Optional: Attribute labels (for analysis, not used in training)

**Prompts**: Automatically generated from masks
- Point prompts: Random sampling from mask regions
- Box prompts: Computed from mask bounding boxes

#### SAM3 Fine-tuning  
**Format**: COCO-style JSON with text prompts
- All SAM2 requirements PLUS:
- **noun_phrase** field for each annotation (required)
- Descriptive text concepts (e.g., "ripe red apple", "damaged fruit")
- Support for negative prompts (phrases with no matching masks)

**Example annotation**:
```json
{
  "id": 10,
  "image_id": 1,
  "segmentation": [[100,150, 300,150, 300,270, 100,270]],
  "bbox": [100, 150, 200, 120],
  "noun_phrase": "ripe red apple"  // Required for SAM3
}
```

### Training Configuration

#### SAM2 Training
**Hyperparameters** (from `configs/train_sam2.yml`):
- Learning rate: 1e-4 with cosine annealing
- Batch size: 2-8 (adjust for GPU memory)
- Loss weights: focal=20.0, dice=1.0, iou=1.0, bce=5.0
- Epochs: 50 (with early stopping)
- Warmup: 2 epochs linear warmup

**Memory Management**:
- Full fine-tuning: 11GB (batch_size=2) to 24GB (batch_size=8)
- Freeze image encoder: Reduces memory by ~30%
- Mixed precision: Enabled by default (2x speedup)

**Trainable Components**:
- Image encoder (optional freeze)
- Prompt encoder (optional freeze)
- Mask decoder (always trained)

#### SAM3 Training
**Hyperparameters** (from `configs/train_sam3.yml`):
- Learning rate: 5e-5 (lower than SAM2 due to pre-training)
- Batch size: 1 with gradient_accumulation=4 (effective batch_size=4)
- Resolution: 1008 (per SAM3 recommendations)
- Loss weights: segmentation=1.0, grounding=0.5, attribute=0.3
- Epochs: 30 (faster convergence than SAM2)

**Memory Management** (Critical):
- Full fine-tuning: ~18GB at batch_size=1, resolution=1008
- Freeze vision encoder: Reduces to ~13GB
- Freeze both encoders: Reduces to ~9GB
- Based on official SAM3 guidelines (GitHub issue #163)

**Trainable Components** (Per SAM3 Design):
- Detector: Always trained ✓
- Shared backbone: Always trained ✓
- Vision encoder: Optional (freeze to save memory)
- Text encoder: Optional (freeze to save memory)
- Tracker modules: Never trained (frozen in eval mode) ✗

### Training Procedures

#### SAM2 Training Loop
1. Load pre-trained SAM2 checkpoint
2. Configure trainable parameters (selective freezing)
3. For each batch:
   - Generate prompts from ground truth masks
   - Forward pass through image encoder → prompt encoder → mask decoder
   - Compute combined loss (focal + dice + IoU + BCE)
   - Backpropagate through trainable layers
   - Clip gradients (max_norm=1.0)
   - Update parameters
4. Validate every epoch on geometric metrics
5. Save best model based on validation IoU

#### SAM3 Training Loop
1. Load pre-trained SAM3 checkpoint
2. Freeze tracker modules (set to eval mode)
3. For each batch:
   - Encode images through vision encoder
   - Encode text prompts through language model
   - Fuse vision-language features
   - Generate masks through detector
   - Compute multi-task loss (segmentation + grounding + attribute)
   - Backpropagate through detector and backbone only
   - Gradient accumulation (4 steps before optimizer update)
   - Clip gradients (max_norm=1.0)
4. Validate on geometric AND semantic metrics
5. Save best model based on validation loss

### Validation During Training

**SAM2 Metrics**:
- Mean IoU (primary)
- Mean Dice coefficient
- Boundary F1 score
- Prompt efficiency (points per mask)

**SAM3 Metrics**:
- Mean IoU (geometric quality)
- Concept recall (semantic understanding)
- Grounding accuracy (text-visual alignment)
- Attribute accuracy (concept reasoning)

### Reproducibility

**Random Seeds**:
- Global seed: 42
- PyTorch seed: 42
- NumPy seed: 42
- Python hash seed: 42

**Hardware Specifications**:
- Tested on NVIDIA A100 (40GB/80GB)
- Compatible with RTX 3090/4090 (24GB)
- Minimum: RTX 3080 (10-12GB) with freezing

**Software Versions**:
- Python: 3.10+
- PyTorch: 2.5+
- CUDA: 12.1+
- SAM2: Latest from facebookresearch/sam2
- SAM3: Latest from facebookresearch/sam3

### Checkpoint Management

**Saving Strategy**:
- Save every 5 epochs
- Always save latest model
- Save best model (lowest validation loss)
- Include optimizer and scheduler state for resumption

**Checkpoint Contents**:
```python
{
  'epoch': current_epoch,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'scheduler_state_dict': scheduler.state_dict(),
  'best_val_loss': best_validation_loss,
  'training_history': metrics_history,
  'config': training_configuration
}
```

### Expected Training Time

**SAM2** (MineApple dataset, ~840 training images):
- A100 GPU: 2-3 hours for 50 epochs
- RTX 3090: 3-4 hours for 50 epochs
- RTX 3080 (with freezing): 4-5 hours for 50 epochs

**SAM3** (Concept-annotated dataset, ~840 training images):
- A100 GPU: 4-6 hours for 30 epochs
- RTX 3090: 6-8 hours for 30 epochs
- RTX 3080 (with freezing): 8-10 hours for 30 epochs

### Post-Training Evaluation

After fine-tuning, models are evaluated using the same protocols as pre-trained inference:
- Run `src/run_experiment.py` with trained checkpoints
- Compare fine-tuned vs pre-trained performance
- Test on held-out test set (never seen during training)
- Validate generalization to unseen concepts (SAM3 only)

For detailed fine-tuning instructions and troubleshooting, see [`docs/training_guide.md`](training_guide.md).

## Conclusion

This methodology establishes a rigorous, reproducible framework for comparing SAM2 and SAM3. By carefully controlling for confounds, using diverse metrics, and ensuring statistical rigor, we enable fair assessment of the architectural gap. The protocols balance ecological validity (realistic use cases) with experimental control (reproducibility).

Our approach provides a template for evaluating future vision-language segmentation models and understanding the evolution of multimodal AI systems.

---

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  
**Last Updated**: 2025  
**Version**: 1.0
