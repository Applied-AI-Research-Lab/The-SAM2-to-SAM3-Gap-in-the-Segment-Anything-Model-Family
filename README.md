# The SAM2-to-SAM3 Gap in the Segment Anything Model Family

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/abs/2512.06032)

## Overview

This repository contains the official implementation and experimental code for our paper:

**"The SAM2-to-SAM3 Gap in the Segment Anything Model Family: Why Prompt-Based Expertise Fails in Concept-Driven Image Segmentation"**

*Ranjan Sapkota¹, Konstantinos I. Roumeliotis², Manoj Karkee¹*

¹Cornell University, Ithaca, NY 14850, USA  
²University of the Peloponnese, Tripoli 22131, Greece

## Citation

If you use this code or our findings in your research, please [cite](https://arxiv.org/abs/2512.06032):

```bibtex
@article{sapkota2025sam2sam3gap,
  title={The SAM2-to-SAM3 Gap in the Segment Anything Model Family: Why Prompt-Based Expertise Fails in Concept-Driven Image Segmentation},
  author={Sapkota, Ranjan and Roumeliotis, Konstantinos I. and Karkee, Manoj},
  journal={arXiv preprint 	arXiv:2512.06032},
  doi={10.48550/arXiv.2512.06032},
  url={https://arxiv.org/abs/2512.06032},
  year={2025}
}
```

### Abstract

This paper investigates the fundamental discontinuity between SAM2 and SAM3, explaining why expertise in prompt-based segmentation fails to transfer to SAM3's concept-driven, multimodal paradigm. SAM2 operates through spatial prompts—points, boxes, and masks—yielding purely geometric and temporal segmentation. In contrast, SAM3 introduces a unified vision–language architecture capable of open-vocabulary reasoning, semantic grounding, contrastive alignment, and exemplar-based concept understanding.

We structure this analysis through five core components:

1. **Conceptual Break**: Contrasting SAM2's spatial prompt semantics with SAM3's multimodal fusion
2. **Architectural Divergence**: SAM2's pure vision–temporal design versus SAM3's vision–language integration
3. **Dataset and Annotation Differences**: SA-V video masks versus multimodal concept-annotated corpora
4. **Training and Hyperparameter Distinctions**: Why SAM2 optimization knowledge does not apply to SAM3
5. **Evaluation Metrics**: Transition from geometric IoU metrics to semantic, open-vocabulary evaluation

## Key Contributions

- Comprehensive analysis of the SAM2-to-SAM3 architectural gap
- Experimental validation on the MineApple orchard dataset
- Novel metrics for comparing prompt-based vs concept-driven segmentation
- Practical guidelines for transitioning from SAM2 to SAM3 deployment
- **Complete fine-tuning infrastructure for both SAM2 and SAM3** (NEW!)
- Production-ready training scripts with memory-efficient implementations
- Based on official SAM3 training guidelines (verified against GitHub issue #163)

## Repository Structure

```
sam2-to-sam3-gap/
├── README.md                      # This file
├── LICENSE                        # Apache 2.0 License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
│
├── configs/                       # Experiment configurations
│   ├── sam2_mineapple.yml        # SAM2 inference settings
│   ├── sam3_mineapple.yml        # SAM3 fine-tuned settings
│   ├── eval_sam2.yml             # Geometric evaluation config
│   ├── eval_sam3.yml             # Concept evaluation config
│   ├── train_sam2.yml            # SAM2 training config (NEW!)
│   └── train_sam3.yml            # SAM3 training config (NEW!)
│
├── data/                          # Dataset directory
│   ├── README.md                 # Data download instructions
│   ├── download_mineapple.py     # Automated dataset downloader
│   └── sample_images/            # Demo images
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── models/                   # Model wrappers
│   │   ├── sam2_wrapper.py      # SAM2 interface
│   │   └── sam3_wrapper.py      # SAM3 interface
│   ├── prompts/                  # Prompt generators
│   │   ├── sam2_prompts.py      # Visual prompts
│   │   └── sam3_text_prompts.py # Text prompts
│   ├── eval/                     # Evaluation metrics
│   │   ├── metrics_geom.py      # Geometric metrics
│   │   └── metrics_concept.py   # Concept metrics
│   ├── utils/                    # Utilities
│   │   ├── dataset_loader.py    # MineApple loader
│   │   ├── visualization.py     # Plotting utilities
│   │   └── logging_utils.py     # Experiment logging
│   ├── training/                 # Fine-tuning infrastructure (NEW!)
│   │   ├── __init__.py
│   │   ├── finetune_sam2.py     # SAM2 fine-tuning script
│   │   └── finetune_sam3.py     # SAM3 fine-tuning script
│   └── run_experiment.py         # Main experiment script
│
├── experiments/                   # Experiment scripts
│   ├── sam2_baseline_mineapple.sh
│   ├── sam3_concept_mineapple.sh
│   └── compare_sam2_sam3.sh
│
├── notebooks/                     # Interactive demos
│   └── 01_sam2_vs_sam3_mineapple.ipynb
│
├── docs/                          # Documentation
│   ├── overview.md               # Conceptual overview
│   ├── methodology.md            # Experimental methodology
│   └── training_guide.md         # Fine-tuning guide (NEW!)
│
└── results/                       # Experimental outputs
    ├── sam2_baseline/            # SAM2 inference results
    ├── sam3_concept/             # SAM3 inference results
    ├── comparison/               # Comparative analysis
    └── training/                 # Fine-tuning checkpoints (NEW!)
        ├── sam2_finetune/
        └── sam3_finetune/
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for SAM2 and SAM3)
- 16GB+ RAM
- 50GB+ disk space for datasets and checkpoints

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Applied-AI-Research-Lab/The-SAM2-to-SAM3-Gap-in-the-Segment-Anything-Model-Family.git
cd The-SAM2-to-SAM3-Gap-in-the-Segment-Anything-Model-Family
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download the MineApple dataset:**
```bash
python data/download_mineapple.py
```

5. **Download SAM2 and SAM3 checkpoints:**
```bash
# SAM2 checkpoints
mkdir -p checkpoints/sam2
cd checkpoints/sam2
./download_sam2_checkpoints.sh

# SAM3 checkpoints (requires Hugging Face authentication)
cd ../sam3
huggingface-cli login
./download_sam3_checkpoints.sh
```

## Quick Start

### 1. Inference (Pre-trained Models)

#### Running SAM2 Baseline

```bash
# Run SAM2 with visual prompts on MineApple
bash experiments/sam2_baseline_mineapple.sh
```

#### Running SAM3 with Text Prompts

```bash
# Run SAM3 with concept-level prompts
bash experiments/sam3_concept_mineapple.sh
```

#### Comparing SAM2 and SAM3

```bash
# Run both models and generate comparison metrics
bash experiments/compare_sam2_sam3.sh
```

### 2. Fine-tuning (NEW!)

#### SAM2 Fine-tuning (Geometric Prompts)

```bash
# Fine-tune SAM2 on your custom dataset
python src/training/finetune_sam2.py --config configs/train_sam2.yml

# Resume from checkpoint
python src/training/finetune_sam2.py \
    --config configs/train_sam2.yml \
    --resume results/training/sam2_finetune/latest_model.pth
```

#### SAM3 Fine-tuning (Text Prompts)

```bash
# Fine-tune SAM3 with vision-language training
python src/training/finetune_sam3.py --config configs/train_sam3.yml

# Resume from checkpoint
python src/training/finetune_sam3.py \
    --config configs/train_sam3.yml \
    --resume results/training/sam3_finetune/latest_model.pth
```

**📚 See [Training Guide](docs/training_guide.md) for comprehensive fine-tuning instructions!**

### 3. Interactive Demo

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_sam2_vs_sam3_mineapple.ipynb
```

## Features

### 🔬 Inference & Evaluation
- ✅ Pre-trained SAM2 and SAM3 model wrappers
- ✅ Geometric prompt generation (points, boxes, masks)
- ✅ Text prompt templates for concept-driven segmentation
- ✅ Comprehensive metrics: geometric (IoU, Dice, Boundary F1) and semantic (concept recall, grounding accuracy)
- ✅ Side-by-side comparison experiments
- ✅ Visualization tools for qualitative analysis
- ✅ Interactive Jupyter notebook demos

### 🎯 Fine-tuning (NEW!)
- ✅ **SAM2 fine-tuning**: Geometric prompt-based training
  - Multiple loss functions (focal, dice, IoU, BCE)
  - Selective layer freezing for memory efficiency
  - Point and box prompt training
  - Mixed precision training (2x speedup)
  - Memory: 11GB-24GB depending on configuration

- ✅ **SAM3 fine-tuning**: Vision-language concept training
  - Based on official SAM3 training guidelines (GitHub issue #163)
  - COCO-style annotations with text prompts (noun_phrase field)
  - Trains detector + backbone, freezes tracker (per SAM3 design)
  - Concept-driven losses: segmentation + grounding + attributes
  - Memory-efficient: ~18GB at batch_size=1, resolution=1008
  - Gradient accumulation for larger effective batch sizes

### 📚 Documentation
- ✅ Comprehensive training guide with troubleshooting
- ✅ Dataset preparation instructions
- ✅ Memory management strategies
- ✅ Conceptual overview of the SAM2-to-SAM3 gap
- ✅ Rigorous experimental methodology

## Experimental Validation

Our experiments use the **MineApple** dataset, which contains orchard images with annotated apple instances. This dataset is ideal for comparing prompt-based and concept-driven segmentation because:

1. **Spatial complexity**: Apples appear at varying distances, occlusions, and lighting conditions
2. **Semantic attributes**: Distinguishing ripe vs. unripe, healthy vs. damaged fruit
3. **Real-world relevance**: Applicable to precision agriculture and automated harvesting

### Metrics

**SAM2 Geometric Metrics:**
- Intersection over Union (IoU)
- Boundary F1 Score
- Temporal Stability (for video sequences)
- Clicks-per-image efficiency

**SAM3 Concept Metrics:**
- Concept Recall (proportion of target instances detected)
- Semantic Localization Error
- Open-Vocabulary F1 Score
- Attribute Segmentation Accuracy

## Results Summary

### Inference (Pre-trained Models)
Our experiments demonstrate:

1. **Prompt Efficiency**: SAM2 requires manual prompts for each apple instance, while SAM3 segments all instances from a single text prompt
2. **Semantic Understanding**: SAM3 successfully distinguishes "ripe apples" from "unripe apples" without spatial guidance
3. **Generalization**: SAM3 shows superior zero-shot performance on unseen concept variations
4. **Failure Modes**: SAM2 fails on ambiguous spatial regions; SAM3 struggles with highly polysemous language

### Fine-tuning (Custom Datasets)
Expected improvements after fine-tuning:

**SAM2 (50 epochs on MineApple):**
- Mean IoU: 0.85-0.92 (vs 0.80-0.88 pre-trained)
- Boundary F1: 0.88-0.94 (vs 0.82-0.90 pre-trained)
- Training time: 2-3 hours on A100 GPU

**SAM3 (30 epochs on concept-annotated data):**
- Mean IoU: 0.88-0.94 (matches SAM2 geometric accuracy)
- Concept Recall: 0.82-0.90 (semantic understanding)
- Grounding Accuracy: 0.75-0.85 (text-visual alignment)
- Training time: 4-6 hours on A100 GPU

Detailed results are available in the `results/` directory and the paper.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for releasing SAM2 and SAM3
- Cornell University for computational resources
- University of the Peloponnese for research support
- The MineApple dataset curators at University of Minnesota

## Links

- [arXiv Paper](https://arxiv.org/abs/2512.06032)
- [SAM2 GitHub](https://github.com/facebookresearch/sam2)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [MineApple Dataset](https://conservancy.umn.edu/)

## Keywords
Segment Anything Model, SAM3, SAM2, SAMv3, SAMv2, Segment with Concepts

