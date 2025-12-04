# MineApple Dataset

## Overview

The **MineApple** dataset is a comprehensive collection of orchard images containing annotated apple instances. This dataset is provided by the University of Minnesota and is designed for precision agriculture research, particularly for fruit detection and segmentation tasks.

## Dataset Information

- **Source**: University of Minnesota Conservancy
- **URL**: https://conservancy.umn.edu/
- **Direct Download**: https://conservancy.umn.edu/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/download
- **Size**: Approximately 2.5 GB (compressed)
- **Format**: JPEG images with JSON annotations
- **License**: Please refer to the dataset webpage for licensing information

## Dataset Structure

After downloading and extracting, the dataset will have the following structure:

```
data/
├── mineapple_raw/                  # Raw downloaded data
│   ├── images/                     # Raw orchard images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/                # Instance annotations
│       ├── train_annotations.json
│       ├── val_annotations.json
│       └── test_annotations.json
│
├── mineapple_processed/            # Processed data for experiments
│   ├── images/                     # Preprocessed images
│   ├── masks/                      # Ground truth masks
│   └── metadata/                   # Image metadata and splits
│
└── sample_images/                  # Demo images (5-10 images)
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

## Downloading the Dataset

### Automated Download

Use the provided download script:

```bash
python data/download_mineapple.py
```

This script will:
1. Download the dataset from the University of Minnesota repository
2. Verify the download integrity
3. Extract the archive
4. Organize files into the expected structure
5. Generate processed versions for experiments

### Manual Download

If you prefer to download manually:

1. Visit: https://conservancy.umn.edu/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/download
2. Download the dataset archive
3. Extract to `data/mineapple_raw/`
4. Run preprocessing: `python data/preprocess_mineapple.py`

## Dataset Statistics

- **Total Images**: ~1,200
- **Training Images**: ~800
- **Validation Images**: ~200
- **Test Images**: ~200
- **Total Apple Instances**: ~15,000+
- **Images per Row**: Variable
- **Average Apples per Image**: ~12.5
- **Image Resolution**: 1920x1080 pixels (typical)

## Annotation Format

Annotations are provided in JSON format following the COCO annotation structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "orchard_001.jpg",
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234.5,
      "bbox": [x, y, width, height],
      "iscrowd": 0,
      "attributes": {
        "ripeness": "ripe",
        "health": "healthy",
        "color": "red"
      }
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "apple",
      "supercategory": "fruit"
    }
  ]
}
```

## Attribute Labels

The dataset includes the following attribute annotations:

### Ripeness
- `ripe`: Fully mature, ready for harvest
- `unripe`: Not yet mature, green color dominant
- `partially_ripe`: Transitional stage

### Health Status
- `healthy`: No visible damage or disease
- `damaged`: Physical damage (bruising, cuts, etc.)
- `diseased`: Shows signs of disease (spots, rot, etc.)

### Color
- `red`: Predominantly red color
- `green`: Green or yellow-green
- `yellow`: Yellow or gold color

## Usage in Experiments

### SAM2 Experiments

SAM2 requires visual prompts (points, boxes, masks). For this dataset:
- Use bounding box annotations as initial prompts
- Generate click points from segmentation centroids
- Evaluate geometric accuracy (IoU, boundary F1)

### SAM3 Experiments

SAM3 uses text prompts for concept-driven segmentation:
- Text prompts: "ripe apples", "all apples", "healthy apples"
- Attribute-based prompts: "red apples", "damaged apples"
- Open-vocabulary prompts: "clustered fruit", "occluded apples"
- Evaluate concept recall and semantic grounding

## Data Preprocessing

The preprocessing script performs:

1. **Image Resizing**: Standardize to common sizes (1024x1024 for SAM2, 1008x1008 for SAM3)
2. **Normalization**: Apply appropriate mean/std normalization
3. **Mask Generation**: Convert polygon annotations to binary masks
4. **Train/Val/Test Split**: Create consistent splits
5. **Metadata Extraction**: Extract image statistics and attributes

Run preprocessing:
```bash
python data/preprocess_mineapple.py --input data/mineapple_raw --output data/mineapple_processed
```

## Sample Images

The `sample_images/` directory contains 5-10 representative images for quick testing and demonstration. These are intentionally small files suitable for version control and demos.

## Citation

If you use the MineApple dataset in your research, please cite:

```bibtex
@dataset{mineapple2023,
  title={MineApple: An Orchard Apple Detection and Segmentation Dataset},
  author={University of Minnesota},
  year={2023},
  publisher={University of Minnesota Digital Conservancy},
  url={https://conservancy.umn.edu/}
}
```

## Troubleshooting

### Download Issues

If the download fails:
- Check your internet connection
- Verify the URL is still active
- Try manual download
- Contact the dataset maintainers

### Extraction Issues

If extraction fails:
- Verify downloaded file integrity (checksum)
- Ensure sufficient disk space (~10 GB free)
- Check file permissions

### Processing Issues

If preprocessing fails:
- Ensure required packages are installed (`pip install -r requirements.txt`)
- Check input data structure matches expected format
- Review error logs in `logs/preprocessing.log`

## Support

For issues with:
- **Dataset content**: Contact University of Minnesota
- **Download script**: Open an issue in this repository
- **Usage in experiments**: See main README.md or contact authors

## Acknowledgments

We thank the University of Minnesota for making this dataset publicly available for research purposes.
