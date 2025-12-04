# Fine-tuning Guide for SAM2 and SAM3

This guide provides comprehensive instructions for fine-tuning both SAM2 and SAM3 models on custom datasets.

## Table of Contents

- [Overview](#overview)
- [SAM2 Fine-tuning](#sam2-fine-tuning)
- [SAM3 Fine-tuning](#sam3-fine-tuning)
- [Dataset Preparation](#dataset-preparation)
- [Training Tips](#training-tips)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is Fine-tuning?

Fine-tuning adapts pre-trained models to your specific dataset and task. This codebase provides complete training infrastructure for both:

- **SAM2**: Prompt-based geometric segmentation (points, boxes, masks)
- **SAM3**: Concept-driven semantic segmentation (text prompts, natural language)

### When to Fine-tune vs Inference-Only

✅ **Fine-tune when:**
- Your domain differs significantly from training data (e.g., specialized crops, rare objects)
- You need improved accuracy on specific object types
- You have labeled data available
- Inference performance is insufficient

❌ **Use pre-trained when:**
- General-purpose segmentation is sufficient
- Limited training data (<100 images)
- Quick prototyping or evaluation

---

## SAM2 Fine-tuning

### Quick Start

```bash
# 1. Prepare your dataset in MineApple format
python data/download_mineapple.py  # or organize your own data

# 2. Configure training
vim configs/train_sam2.yml  # adjust hyperparameters

# 3. Start training
python src/training/finetune_sam2.py --config configs/train_sam2.yml

# 4. Resume from checkpoint (optional)
python src/training/finetune_sam2.py --config configs/train_sam2.yml --resume results/training/sam2_finetune/latest_model.pth
```

### Configuration Guide

Key parameters in `configs/train_sam2.yml`:

```yaml
model:
  type: "vit_h"  # vit_h (huge, 224M), vit_l (large), vit_b (base)
  checkpoint: "checkpoints/sam2_hiera_large.pt"
  freeze_image_encoder: false  # Set true to save memory

training:
  num_epochs: 50
  batch_size: 2  # Adjust based on GPU: 8 (24GB), 4 (16GB), 2 (11GB)
  learning_rate: 1.0e-4
  prompt_type: "point"  # "point" or "box"

loss:
  focal: 20.0   # Class imbalance handling
  dice: 1.0     # Overlap optimization
  iou: 1.0      # IoU prediction
  bce: 5.0      # Binary cross-entropy
```

### Memory Management

| Configuration | GPU Memory | Batch Size | Training Speed |
|--------------|------------|------------|----------------|
| Full fine-tuning | ~22GB | 8 | Fast |
| Full fine-tuning | ~11GB | 2 | Medium |
| Freeze image encoder | ~8GB | 2 | Medium |
| Freeze image encoder | ~15GB | 4 | Medium-Fast |

**Recommendations:**
- RTX 3090/4090 (24GB): `batch_size=8`, full fine-tuning
- RTX 3080 (10-12GB): `batch_size=2`, full fine-tuning
- RTX 3070 (8GB): `batch_size=2`, freeze image encoder

### Prompt Types

**Point Prompts:**
- Samples random points from mask regions
- Better generalization to unseen objects
- Simulates interactive annotation
- **Recommended for most use cases**

**Box Prompts:**
- Uses bounding boxes from masks
- Faster convergence
- Better for well-defined objects
- Good for detection + segmentation tasks

### Training Workflow

```python
# The training script handles everything automatically:
# 1. Load pretrained SAM2 checkpoint
# 2. Configure trainable layers (selective freezing)
# 3. Generate prompts from ground truth masks
# 4. Train with combined loss (focal + dice + IoU)
# 5. Validate and save checkpoints
# 6. Track metrics and visualizations
```

### Expected Results

After 50 epochs on MineApple dataset:
- **Mean IoU**: 0.85-0.92 (vs 0.80-0.88 pre-trained)
- **Boundary F1**: 0.88-0.94 (vs 0.82-0.90 pre-trained)
- **Training time**: 2-3 hours on A100 GPU

---

## SAM3 Fine-tuning

### Quick Start

```bash
# 1. Prepare COCO-format data with text prompts
python scripts/convert_to_coco_with_text.py  # convert your data

# 2. Configure training
vim configs/train_sam3.yml

# 3. Start training
python src/training/finetune_sam3.py --config configs/train_sam3.yml

# 4. Resume if needed
python src/training/finetune_sam3.py --config configs/train_sam3.yml --resume results/training/sam3_finetune/latest_model.pth
```

### Configuration Guide

Key parameters in `configs/train_sam3.yml`:

```yaml
model:
  checkpoint: "checkpoints/sam3_hiera_l_coco_sav_vitl14_internvid10m.pt"
  freeze_vision_encoder: false  # Freeze to save memory
  freeze_text_encoder: false
  freeze_tracker: true  # Always true (per SAM3 design)

training:
  num_epochs: 30
  batch_size: 1  # Start with 1 (~18GB memory)
  resolution: 1008  # SAM3 recommended resolution
  learning_rate: 5.0e-5  # Lower than SAM2
  gradient_accumulation: 4  # Simulate batch_size=4

loss:
  segmentation: 1.0  # Mask quality
  grounding: 0.5     # Text-visual alignment
  attribute: 0.3     # Semantic attributes
```

### Memory Management (Critical for SAM3)

| Configuration | GPU Memory | Effective Batch | Training Speed |
|--------------|------------|-----------------|----------------|
| Full fine-tuning | ~18GB | 1 | Slow |
| Full fine-tuning + gradient accumulation (4x) | ~18GB | 4 (effective) | Medium |
| Freeze vision encoder | ~13GB | 1 | Medium |
| Freeze both encoders | ~9GB | 1 | Fast |

**Per official SAM3 guidelines (issue #163):**
- Batch size 1, resolution 1008: **~18GB GPU memory**
- Use gradient accumulation to simulate larger batches
- Freeze encoders if memory constrained

**Recommendations:**
- A100 (40GB/80GB): `batch_size=2`, full fine-tuning
- RTX 3090/4090 (24GB): `batch_size=1`, gradient accumulation 4x
- RTX 3080 (10-12GB): Freeze vision encoder, gradient accumulation 2x

### What Gets Trained (Per SAM3 Official Guidelines)

✅ **Always Trained:**
- Detector (segmentation head)
- Shared backbone (feature extraction)

❌ **Never Trained:**
- Tracker modules (inference mode only)

🔄 **Optional (freeze to save memory):**
- Vision encoder (~30% memory savings)
- Text encoder (~20% memory savings)

### Dataset Format (COCO-style with Text)

SAM3 requires **noun_phrase** field for each annotation:

```json
{
  "images": [
    {"id": 1, "file_name": "img1.jpg", "width": 1280, "height": 720}
  ],
  "annotations": [
    {
      "id": 10,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 120],
      "segmentation": [[100,150, 300,150, 300,270, 100,270]],
      "area": 24000,
      "iscrowd": 0,
      "noun_phrase": "ripe red apple"  // Required: text concept
    }
  ],
  "categories": [{"id": 1, "name": "object"}]
}
```

**Text Prompt Guidelines:**
- Be descriptive: "ripe red apple" > "apple"
- Include attributes: color, ripeness, health, size
- Use natural language: "damaged green fruit"
- Support negatives: phrases with no matching masks (for robustness)

### Training Workflow

```python
# SAM3 training automatically:
# 1. Load pretrained SAM3 checkpoint (848M params)
# 2. Freeze tracker modules (per SAM3 design)
# 3. Encode images through vision encoder
# 4. Encode text prompts through language model
# 5. Fuse vision-language features
# 6. Compute segmentation + grounding loss
# 7. Backpropagate through detector + backbone only
# 8. Validate on concept recall and geometric metrics
```

### Expected Results

After 30 epochs on concept-annotated dataset:
- **Mean IoU**: 0.88-0.94 (competitive with SAM2)
- **Concept Recall**: 0.82-0.90 (measures semantic understanding)
- **Grounding Accuracy**: 0.75-0.85 (text-visual alignment)
- **Training time**: 4-6 hours on A100 GPU

---

## Dataset Preparation

### For SAM2 (MineApple Format)

```
data/mineapple/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── annotations/
│       ├── img_001.json
│       └── ...
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

**Annotation format (JSON per image):**

```json
{
  "image_id": "img_001",
  "width": 1280,
  "height": 720,
  "instances": [
    {
      "instance_id": 1,
      "label": "apple",
      "segmentation": {
        "polygon": [[x1,y1, x2,y2, ...]]
      },
      "attributes": {
        "ripeness": "ripe",
        "color": "red",
        "health": "healthy"
      }
    }
  ]
}
```

### For SAM3 (COCO Format with Text)

```
data/your_dataset/
├── train_images/
│   ├── img001.jpg
│   └── ...
├── val_images/
│   └── ...
└── annotations/
    ├── train_coco.json
    └── val_coco.json
```

Use conversion script:

```bash
# Convert MineApple format to COCO with text prompts
python scripts/convert_mineapple_to_coco_text.py \
    --input data/mineapple \
    --output data/mineapple_coco \
    --attribute-templates configs/text_prompt_templates.yaml
```

---

## Training Tips

### General Best Practices

1. **Start Small:**
   - Test on subset (100 images) first
   - Verify training loop works
   - Check GPU memory usage
   - Then scale to full dataset

2. **Monitor Training:**
   ```bash
   # Watch training progress
   tail -f results/training/sam2_finetune/training.log
   
   # Check GPU usage
   watch -n 1 nvidia-smi
   ```

3. **Learning Rate:**
   - SAM2: Start with 1e-4
   - SAM3: Start with 5e-5 (lower due to pre-training)
   - Reduce by 10x if loss plateaus
   - Use warmup (2 epochs) for stability

4. **Data Augmentation:**
   - Enable horizontal flip (0.5)
   - Moderate color jitter (0.2)
   - Avoid aggressive augmentation (models are robust)

5. **Validation:**
   - Validate every epoch
   - Track both training and validation loss
   - Early stopping if val loss plateaus (10 epochs patience)

### SAM2-Specific Tips

- **Point prompts** generalize better than boxes
- Use **multi-point training** (3-5 points per mask) for better prompt robustness
- **Freeze image encoder** if memory constrained (minimal accuracy loss)
- Train for **50-100 epochs** (more data = fewer epochs needed)

### SAM3-Specific Tips (From Official Guidelines)

- **Batch size 1** is normal due to memory (use gradient accumulation)
- Always **freeze tracker** (it's not designed for training)
- Include **negative text prompts** (20% ratio) for robustness
- **Text quality matters**: descriptive phrases > single words
- Monitor **grounding loss**: should decrease steadily
- **Evaluate on unseen concepts** to test open-vocabulary capability

---

## Troubleshooting

### Out of Memory (OOM) Errors

**SAM2:**
```yaml
# Try these in order:
1. Reduce batch_size: 8 -> 4 -> 2
2. Enable freeze_image_encoder: true
3. Reduce image resolution (1024 -> 512)
4. Disable mixed_precision (slower but less memory)
```

**SAM3:**
```yaml
# More aggressive (model is larger):
1. Start with batch_size: 1
2. Enable gradient_accumulation: 4
3. Freeze vision encoder: freeze_vision_encoder: true
4. Freeze text encoder: freeze_text_encoder: true
5. Reduce resolution: 1008 -> 720
```

### Loss Not Decreasing

**Check:**
1. Learning rate too high (reduce 10x)
2. Data loading correctly (visualize batch)
3. Loss weights balanced (focal:dice:iou = 20:1:1)
4. Gradient clipping not too aggressive (try 5.0)

**SAM3 specific:**
- Verify noun_phrase field exists in annotations
- Check text encoder is not frozen (unless intentional)
- Increase grounding loss weight (0.5 -> 1.0)

### Poor Generalization

**Symptoms:**
- Training loss low, validation loss high
- Overfitting to training set

**Solutions:**
1. Enable data augmentation
2. Reduce num_epochs (50 -> 30)
3. Increase weight_decay (0.01 -> 0.05)
4. Add more training data
5. SAM3: Include more negative prompts

### Slow Training

**Speed up:**
1. Enable mixed_precision: true (2x faster)
2. Increase num_workers: 4 -> 8 (faster data loading)
3. Enable pin_memory: true
4. Use smaller model (vit_l instead of vit_h)
5. Reduce resolution (minimal accuracy loss)

### Checkpoint Not Loading

**Common issues:**
```python
# Error: "Unexpected key in state_dict"
# Solution: Use strict=False
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Error: "Checkpoint not found"
# Solution: Download from official repos
# SAM2: https://github.com/facebookresearch/segment-anything-2
# SAM3: https://github.com/facebookresearch/sam3
```

---

## Validation and Evaluation

### During Training

Both scripts automatically:
- Validate after each epoch
- Compute metrics (IoU, Dice, Boundary F1)
- Save best model based on validation loss
- Generate visualizations

### Post-Training Evaluation

```bash
# SAM2 evaluation
python src/run_experiment.py \
    --mode sam2 \
    --config configs/eval_sam2.yml \
    --checkpoint results/training/sam2_finetune/best_model.pth

# SAM3 evaluation
python src/run_experiment.py \
    --mode sam3 \
    --config configs/eval_sam3.yml \
    --checkpoint results/training/sam3_finetune/best_model.pth

# Compare with baseline
python src/run_experiment.py --mode compare
```

### Metrics to Watch

**SAM2 (Geometric):**
- Mean IoU: >0.85 is good, >0.90 is excellent
- Boundary F1: >0.85 indicates precise boundaries
- Mean Dice: Similar to IoU, redundant but commonly reported

**SAM3 (Geometric + Semantic):**
- Mean IoU: Should match or exceed SAM2
- Concept Recall: >0.80 shows good semantic understanding
- Grounding Accuracy: >0.75 indicates proper text-visual alignment
- Attribute Accuracy: Domain-specific, aim for >0.70

---

## Advanced Topics

### Multi-GPU Training

```python
# Wrap model with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Adjust batch size accordingly
batch_size = base_batch_size * num_gpus
```

### Custom Loss Functions

Edit `compute_loss()` methods in:
- `src/training/finetune_sam2.py`
- `src/training/finetune_sam3.py`

Add your loss and adjust weights in config YAML.

### Transfer Learning

```python
# Fine-tune on Dataset A, then adapt to Dataset B
# 1. Train on Dataset A
python src/training/finetune_sam2.py --config configs/train_dataset_a.yml

# 2. Resume from Dataset A checkpoint for Dataset B
python src/training/finetune_sam2.py \
    --config configs/train_dataset_b.yml \
    --resume results/training/dataset_a/best_model.pth
```

---

## References

### Official Documentation

- **SAM2**: https://github.com/facebookresearch/segment-anything-2
- **SAM3**: https://github.com/facebookresearch/sam3
- **SAM3 Training Guide**: https://github.com/facebookresearch/sam3/issues/163

### Papers

- SAM2: "Segment Anything in Images and Videos" (Ravi et al., 2024)
- SAM3: "Segment Anything with Concepts" (Zou et al., 2025)
- This work: "The SAM2-to-SAM3 Gap in the Segment Anything Model Family" (Sapkota, Roumeliotis, Karkee, 2025)

---

## Support

For issues or questions:
1. Check this guide and configuration files
2. Review troubleshooting section
3. Consult official SAM2/SAM3 documentation
4. Open an issue on this repository

**Happy Training! 🚀**
