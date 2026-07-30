#!/usr/bin/env python3

import csv
from pathlib import Path
import sys

def create_symlinks_from_csv(csv_file, dataset_dir):
    """
    Create symbolic links based on the CSV mapping.
    Skip any image that has missing masks (FALSE in any mask column).
    """
    
    # Create target directories
    dataset_path = Path(dataset_dir)
    train_images = dataset_path / "train_images"
    train_masks = dataset_path / "train_masks"
    val_images = dataset_path / "val_images"
    val_masks = dataset_path / "val_masks"
    
    for dir_path in [train_images, train_masks, val_images, val_masks]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total_rows': 0,
        'skipped_missing_masks': 0,
        'train_images_linked': 0,
        'train_masks_linked': 0,
        'val_images_linked': 0,
        'val_masks_linked': 0
    }
    
    # Read CSV and create links
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total_rows'] += 1
            
            image_path = Path(row['image_path'])
            root_mask_path = row['root_mask_path']
            shoot_mask_path = row['shoot_mask_path']
            seed_mask_path = row['seed_mask_path']
            
            # Skip if any mask is missing
            if 'FALSE' in [root_mask_path, shoot_mask_path, seed_mask_path]:
                stats['skipped_missing_masks'] += 1
                continue
            
            # Determine if this is a validation image (from Y2B_25 and starts with val_)
            is_val = 'Y2B_25' in str(image_path) and image_path.name.startswith('val_')
            
            # Create image symlink
            if is_val:
                target_image = val_images / image_path.name
                target_mask_dir = val_masks
                stats['val_images_linked'] += 1
            else:
                target_image = train_images / image_path.name
                target_mask_dir = train_masks
                stats['train_images_linked'] += 1
            
            # Create image link (remove if exists)
            if target_image.exists() or target_image.is_symlink():
                target_image.unlink()
            target_image.symlink_to(image_path.absolute())
            
            # Create mask symlinks
            for mask_path_str in [root_mask_path, shoot_mask_path, seed_mask_path]:
                mask_path = Path(mask_path_str)
                target_mask = target_mask_dir / mask_path.name
                
                # Remove if exists (including broken symlinks)
                if target_mask.exists() or target_mask.is_symlink():
                    target_mask.unlink()
                target_mask.symlink_to(mask_path.absolute())
                
                if is_val:
                    stats['val_masks_linked'] += 1
                else:
                    stats['train_masks_linked'] += 1
    
    return stats

def main():
    csv_file = "image_mask_mapping.csv"
    dataset_dir = "dataset"
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        print("Please run check_masks.py first.")
        sys.exit(1)
    
    print(f"Creating symbolic links from {csv_file}...")
    print(f"Target directory: {dataset_dir}/")
    print()
    
    stats = create_symlinks_from_csv(csv_file, dataset_dir)
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total images processed: {stats['total_rows']}")
    print(f"Skipped (missing masks): {stats['skipped_missing_masks']}")
    print()
    print(f"Training set:")
    print(f"  Images linked: {stats['train_images_linked']}")
    print(f"  Masks linked: {stats['train_masks_linked']}")
    print()
    print(f"Validation set:")
    print(f"  Images linked: {stats['val_images_linked']}")
    print(f"  Masks linked: {stats['val_masks_linked']}")
    print()
    print(f"Total symlinks created: {stats['train_images_linked'] + stats['train_masks_linked'] + stats['val_images_linked'] + stats['val_masks_linked']}")

if __name__ == "__main__":
    main()