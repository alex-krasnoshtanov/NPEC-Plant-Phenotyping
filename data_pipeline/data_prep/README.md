# Plant Root Dataset Organization

Scripts for organizing plant root microscopy images and masks into structured datasets with symbolic links.

## Overview

These scripts process raw microscopy data (Y2B_23, Y2B_24, Y2B_25) and create organized datasets by:
1. Mapping images to their corresponding masks (root, shoot, seed)
2. Filtering out incomplete images (missing any masks)
3. Creating symbolic links in structured train/val directories

## Prerequisites

- Python 3.7+
- Standard library only (no additional packages needed)

## Directory Structure

Your raw data should be organized as:
```
data/
└── raw/
    ├── Y2B_23/
    │   ├── images/
    │   │   ├── test/
    │   │   └── train/
    │   └── masks/
    ├── Y2B_24/
    │   ├── images/
    │   └── masks/
    │       ├── Alican/
    │       ├── Dean/
    │       ├── Elavendan/
    │       ├── Jason/
    │       ├── Karna/
    │       ├── Myrthe/
    │       └── Shival/
    └── Y2B_25/
        ├── images/
        └── masks/
            ├── Alican/
            ├── Dean/
            ├── Jason/
            ├── Karna/
            ├── Myrthe/
            └── Uther/
```

## Quick Start

### Step 1: Generate Image-Mask Mapping
```bash
python3 check_masks.py
```

This creates `image_mask_mapping.csv` with paths to all images and their corresponding masks. Images missing any of the three required masks (root, shoot, seed) are marked with "FALSE".

**Output**: 
- CSV with 4 columns: image_path, root_mask_path, shoot_mask_path, seed_mask_path
- Summary statistics showing complete vs incomplete images

### Step 2: Create Dataset

Choose one of the following dataset configurations:

#### Option A: Default Dataset (dataset/)
Uses Y2B_25 for validation:
```bash
python3 symlink_files.py
```

**Dataset composition**:
- Training: Y2B_23 (all) + Y2B_24 (all) + Y2B_25 (train_*)
- Validation: Y2B_25 (val_*)

#### Option B: Dataset 24 (dataset_24/)
Uses Y2B_24 for validation, excludes Y2B_25:
```bash
python3 symlink_files_dataset_24.py
```

**Dataset composition**:
- Training: Y2B_23 (all) + Y2B_24 (train_*)
- Validation: Y2B_24 (val_*)
- Excluded: All Y2B_25

### Step 3: Verify Results

Count files in each directory:
```bash
find dataset_24/train_images -type l | wc -l
find dataset_24/train_masks -type l | wc -l
find dataset_24/val_images -type l | wc -l
find dataset_24/val_masks -type l | wc -l
```

## Output Structure
```
dataset_24/
├── train_images/     # Symbolic links to training images
├── train_masks/      # Symbolic links to all training masks
├── val_images/       # Symbolic links to validation images
└── val_masks/        # Symbolic links to all validation masks
```

Note: Each image has 3 corresponding masks (root, shoot, seed) in the masks directory.

## Troubleshooting

### Analyze Mismatches
If you have a separate list of PNG files, compare it against the CSV:
```bash
python3 analyze_mismatch.py
```

This requires a `png.csv` file with one filepath per line.

### Common Issues

**Missing masks**: Some images don't have all three required masks. These are automatically skipped during dataset creation.

**File counts**: 
- Images: One symlink per source image
- Masks: Three symlinks per source image (root + shoot + seed)
- Expected ratio: masks = images × 3

## Scripts Reference

- **check_masks.py**: Scans raw data and generates CSV mapping
- **symlink_files.py**: Creates default dataset with Y2B_25 validation
- **symlink_files_dataset_24.py**: Creates dataset_24 (Y2B_24 validation only)
- **analyze_mismatch.py**: Diagnostic tool for comparing file lists

## Notes

- Scripts use symbolic links, not copies (saves disk space)
- All paths in CSV are absolute paths
- Missing masks are identified by "FALSE" in CSV columns
- Scripts are idempotent (safe to run multiple times)