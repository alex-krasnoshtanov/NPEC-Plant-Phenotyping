#!/usr/bin/env python3

"""
Unified dataset processing script that:
1. Scans all datasets (Y2B_23, Y2B_24, Y2B_25) for images and masks
2. Identifies missing masks
3. Checks for all-zero masks (blank masks with no annotations)
4. Creates train/val split based on percentage (not filename)
5. Copies files to create the split dataset

Design decisions:
- Random split with fixed seed for reproducibility (alternative: stratified by dataset)
- PIL for mask loading (alternative: OpenCV, but PIL is more standard)
- Skip images with missing or all-zero masks to avoid training on invalid data
- Default 80/20 split (configurable via command line)
- Copy files instead of symlinks for Windows compatibility (alternative: symlinks on Linux/Mac)
"""

import csv
from pathlib import Path
import sys
import random
import shutil
from PIL import Image
import numpy as np

def find_mask(image_path, mask_dir, mask_type):
    """
    Find the mask file for a given image and mask type.
    Returns the full path if found, otherwise returns None.
    """
    img_name = image_path.stem
    
    patterns = [
        f"{img_name}-Fish Eye Corrected_{mask_type}.tif",
        f"{img_name}-Fish Eye Corrected_{mask_type}.tiff",
        f"{img_name}_{mask_type}.tif",
        f"{img_name}_{mask_type}.tiff",
    ]
    
    for pattern in patterns:
        mask_path = mask_dir / pattern
        if mask_path.exists():
            return mask_path
    
    return None

def is_mask_all_zero(mask_path):
    """
    Check if a mask is completely empty (all zeros).
    
    Returns:
        True if mask is all zeros, False otherwise
        
    Note: Some masks might be 0/1 format, others 0/255.
    We check if max value is 0 to catch both cases.
    """
    try:
        # Load mask and convert to numpy array
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Check if all values are zero
        # Using max() is faster than checking every pixel
        return mask_array.max() == 0
    except Exception as e:
        print(f"Warning: Could not read mask {mask_path}: {e}")
        return True  # Treat unreadable masks as invalid

def process_datasets(raw_dir):
    """
    Process all datasets and collect image-mask mappings.
    
    Returns:
        List of dicts with keys: image_path, root_mask, shoot_mask, seed_mask, dataset
    """
    raw_path = Path(raw_dir)
    data = []
    
    # Process Y2B_23 - both test and train subdirectories
    y2b_23_images_base = raw_path / "Y2B_23" / "images"
    y2b_23_masks = raw_path / "Y2B_23" / "masks"
    
    for subdir in ["test", "train"]:
        y2b_23_images = y2b_23_images_base / subdir
        if y2b_23_images.exists():
            for img in sorted(y2b_23_images.glob("*.png")):
                data.append({
                    'image_path': img,
                    'root_mask': find_mask(img, y2b_23_masks, "root_mask"),
                    'shoot_mask': find_mask(img, y2b_23_masks, "shoot_mask"),
                    'seed_mask': find_mask(img, y2b_23_masks, "seed_mask"),
                    'dataset': 'Y2B_23'
                })
    
    # Process Y2B_24 - masks organized by person name
    y2b_24_images = raw_path / "Y2B_24" / "images"
    y2b_24_masks = raw_path / "Y2B_24" / "masks"
    
    if y2b_24_images.exists():
        for img in sorted(y2b_24_images.glob("*.png")):
            parts = img.stem.split('_')
            if len(parts) >= 2:
                person = parts[1]
                person_mask_dir = y2b_24_masks / person
                
                data.append({
                    'image_path': img,
                    'root_mask': find_mask(img, person_mask_dir, "root_mask"),
                    'shoot_mask': find_mask(img, person_mask_dir, "shoot_mask"),
                    'seed_mask': find_mask(img, person_mask_dir, "seed_mask"),
                    'dataset': 'Y2B_24'
                })
    
    # Process Y2B_25 - masks organized by person name
    y2b_25_images = raw_path / "Y2B_25" / "images"
    y2b_25_masks = raw_path / "Y2B_25" / "masks"
    
    if y2b_25_images.exists():
        for img in sorted(y2b_25_images.glob("*.png")):
            parts = img.stem.split('_')
            if len(parts) >= 2:
                person = parts[1]
                person_mask_dir = y2b_25_masks / person
                
                data.append({
                    'image_path': img,
                    'root_mask': find_mask(img, person_mask_dir, "root_mask"),
                    'shoot_mask': find_mask(img, person_mask_dir, "shoot_mask"),
                    'seed_mask': find_mask(img, person_mask_dir, "seed_mask"),
                    'dataset': 'Y2B_25'
                })
    
    return data

def validate_and_filter_data(data, mask_type='root_mask'):
    """
    Validate data and filter out images with missing or all-zero masks.
    
    Args:
        data: List of image-mask mappings
        mask_type: Which mask type to validate ('root_mask', 'shoot_mask', or 'seed_mask')
        
    Returns:
        Tuple of (valid_data, stats_dict)
        
    Design choice: We validate only the specified mask_type since you're training
    separate models for root, shoot, and seed segmentation. This allows some
    flexibility if different mask types have different annotation completeness.
    """
    valid_data = []
    stats = {
        'total': len(data),
        'missing_mask': 0,
        'all_zero_mask': 0,
        'valid': 0
    }
    
    print(f"\nValidating {mask_type}...")
    
    for item in data:
        mask_path = item[mask_type]
        
        # Check if mask exists
        if mask_path is None:
            stats['missing_mask'] += 1
            continue
        
        # Check if mask is all zeros
        if is_mask_all_zero(mask_path):
            stats['all_zero_mask'] += 1
            continue
        
        # Valid image-mask pair
        stats['valid'] += 1
        valid_data.append(item)
    
    return valid_data, stats

def create_train_val_split(data, val_ratio=0.2, random_seed=42):
    """
    Split data into train and validation sets based on percentage.
    
    Args:
        data: List of validated image-mask mappings
        val_ratio: Fraction of data to use for validation (default 0.2 = 20%)
        random_seed: Seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
        
    Alternative approaches:
    - Stratified split by dataset (ensure each dataset is proportionally represented)
    - Stratified split by person (for Y2B_24/25)
    - Time-based split (use earlier data for train, later for val)
    
    Current approach uses simple random split for maximum flexibility.
    """
    # Shuffle data with fixed seed for reproducibility
    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    val_size = int(len(shuffled_data) * val_ratio)
    
    val_data = shuffled_data[:val_size]
    train_data = shuffled_data[val_size:]
    
    return train_data, val_data

def create_dataset(train_data, val_data, dataset_dir, mask_type='root_mask'):
    """
    Copy files to create train and validation sets.
    
    Args:
        train_data: List of training image-mask mappings
        val_data: List of validation image-mask mappings
        dataset_dir: Target directory for the dataset
        mask_type: Which mask type to copy
        
    Note: Using copy instead of symlinks for Windows compatibility.
    On Linux/Mac, symlinks would save disk space, but copies work everywhere.
    """
    dataset_path = Path(dataset_dir)
    train_images = dataset_path / "train_images"
    train_masks = dataset_path / "train_masks"
    val_images = dataset_path / "val_images"
    val_masks = dataset_path / "val_masks"
    
    # Create directories
    for dir_path in [train_images, train_masks, val_images, val_masks]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    def copy_data(data, img_dir, mask_dir):
        """Helper function to copy files for a dataset split."""
        for item in data:
            image_path = item['image_path']
            mask_path = item[mask_type]
            
            # Copy image
            target_image = img_dir / image_path.name
            shutil.copy2(image_path, target_image)
            
            # Copy mask
            target_mask = mask_dir / mask_path.name
            shutil.copy2(mask_path, target_mask)
    
    print("\nCopying files...")
    copy_data(train_data, train_images, train_masks)
    copy_data(val_data, val_images, val_masks)
    
    return {
        'train_images': len(train_data),
        'train_masks': len(train_data),
        'val_images': len(val_data),
        'val_masks': len(val_data)
    }

def print_dataset_distribution(data, label):
    """Print the distribution of images across datasets."""
    from collections import Counter
    dist = Counter(item['dataset'] for item in data)
    print(f"\n{label} distribution:")
    for dataset, count in sorted(dist.items()):
        print(f"  {dataset}: {count} images")

def main():
    """
    Main execution flow:
    1. Locate raw data directory
    2. Process all datasets
    3. Validate and filter data
    4. Create train/val split
    5. Generate symlinks
    """
    # Configuration
    mask_type = 'root_mask'  # Change to 'shoot_mask' or 'seed_mask' as needed
    val_ratio = 0.2  # 20% validation, 80% training
    random_seed = 42
    
    # Paths
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parent.parent / "data" / "raw"
    dataset_dir = script_dir.parent.parent / "data" / "processed" / "dataset_unified"
    
    print("="*60)
    print("Unified Dataset Processing")
    print("="*60)
    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {dataset_dir}")
    print(f"Mask type: {mask_type}")
    print(f"Val ratio: {val_ratio:.1%}")
    print(f"Random seed: {random_seed}")
    
    # Step 1: Process all datasets
    print("\n" + "="*60)
    print("Step 1: Processing datasets...")
    print("="*60)
    data = process_datasets(raw_dir)
    print(f"Found {len(data)} images across all datasets")
    
    # Step 2: Validate and filter
    print("\n" + "="*60)
    print("Step 2: Validating masks...")
    print("="*60)
    valid_data, stats = validate_and_filter_data(data, mask_type)
    
    print(f"\nValidation results:")
    print(f"  Total images: {stats['total']}")
    print(f"  Missing masks: {stats['missing_mask']}")
    print(f"  All-zero masks: {stats['all_zero_mask']}")
    print(f"  Valid images: {stats['valid']}")
    print(f"  Retention rate: {stats['valid']/stats['total']*100:.1f}%")
    
    if stats['valid'] == 0:
        print("\nError: No valid images found!")
        sys.exit(1)
    
    # Step 3: Create train/val split
    print("\n" + "="*60)
    print("Step 3: Creating train/val split...")
    print("="*60)
    train_data, val_data = create_train_val_split(valid_data, val_ratio, random_seed)
    
    print(f"Training set: {len(train_data)} images ({len(train_data)/len(valid_data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} images ({len(val_data)/len(valid_data)*100:.1f}%)")
    
    print_dataset_distribution(train_data, "Training")
    print_dataset_distribution(val_data, "Validation")
    
    # Step 4: Create dataset
    print("\n" + "="*60)
    print("Step 4: Copying files...")
    print("="*60)
    link_stats = create_dataset(train_data, val_data, dataset_dir, mask_type)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset created at: {dataset_dir}")
    print(f"\nFiles copied:")
    print(f"  Training: {link_stats['train_images']} images, {link_stats['train_masks']} masks")
    print(f"  Validation: {link_stats['val_images']} images, {link_stats['val_masks']} masks")
    print(f"  Total: {sum(link_stats.values())} files")
    print("\nDataset is ready for training!")

if __name__ == "__main__":
    main()