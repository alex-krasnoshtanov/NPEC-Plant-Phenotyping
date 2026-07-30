#!/usr/bin/env python3

import csv
from pathlib import Path

def find_mask(image_path, mask_dir, mask_type):
    """
    Find the mask file for a given image and mask type.
    Returns the full path if found, otherwise returns 'FALSE'.
    """
    img_name = image_path.stem  # filename without extension
    
    # Try different mask naming patterns
    patterns = [
        f"{img_name}-Fish Eye Corrected_{mask_type}.tif",
        f"{img_name}-Fish Eye Corrected_{mask_type}.tiff",
        f"{img_name}_{mask_type}.tif",
        f"{img_name}_{mask_type}.tiff",
    ]
    
    for pattern in patterns:
        mask_path = mask_dir / pattern
        if mask_path.exists():
            return str(mask_path.absolute())
    
    return "FALSE"

def process_dataset(raw_dir):
    """Process all datasets and return list of rows for CSV."""
    raw_path = Path(raw_dir)
    rows = []
    
    # Process Y2B_23 - BOTH test AND train directories
    y2b_23_images_base = raw_path / "Y2B_23" / "images"
    y2b_23_masks = raw_path / "Y2B_23" / "masks"
    
    # Process both test and train subdirectories
    for subdir in ["test", "train"]:
        y2b_23_images = y2b_23_images_base / subdir
        if y2b_23_images.exists():
            for img in sorted(y2b_23_images.glob("*.png")):
                rows.append([
                    str(img.absolute()),
                    find_mask(img, y2b_23_masks, "root_mask"),
                    find_mask(img, y2b_23_masks, "shoot_mask"),
                    find_mask(img, y2b_23_masks, "seed_mask")
                ])
    
    # Process Y2B_24
    y2b_24_images = raw_path / "Y2B_24" / "images"
    y2b_24_masks = raw_path / "Y2B_24" / "masks"

    if y2b_24_images.exists():
        for img in sorted(y2b_24_images.glob("*.png")):
            # Extract person name from filename (e.g., train_Alican_212231_im1.png -> Alican)
            parts = img.stem.split('_')
            if len(parts) >= 2:
                person = parts[1]
                person_mask_dir = y2b_24_masks / person
                
                rows.append([
                    str(img.absolute()),
                    find_mask(img, person_mask_dir, "root_mask"),
                    find_mask(img, person_mask_dir, "shoot_mask"),
                    find_mask(img, person_mask_dir, "seed_mask")
                ])
    
    # Process Y2B_25
    y2b_25_images = raw_path / "Y2B_25" / "images"
    y2b_25_masks = raw_path / "Y2B_25" / "masks"

    if y2b_25_images.exists():
        for img in sorted(y2b_25_images.glob("*.png")):
            # Extract person name from filename
            parts = img.stem.split('_')
            if len(parts) >= 2:
                person = parts[1]
                person_mask_dir = y2b_25_masks / person
                
                rows.append([
                    str(img.absolute()),
                    find_mask(img, person_mask_dir, "root_mask"),
                    find_mask(img, person_mask_dir, "shoot_mask"),
                    find_mask(img, person_mask_dir, "seed_mask")
                ])
    
    return rows

def main():
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parent.parent / "data" / "raw"
    output_file = "image_mask_mapping.csv"
    
    print("Processing datasets...")
    rows = process_dataset(raw_dir)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'root_mask_path', 'shoot_mask_path', 'seed_mask_path'])
        writer.writerows(rows)
    
    # Print summary
    total_images = len(rows)
    missing_root = sum(1 for row in rows if row[1] == "FALSE")
    missing_shoot = sum(1 for row in rows if row[2] == "FALSE")
    missing_seed = sum(1 for row in rows if row[3] == "FALSE")
    images_with_missing = sum(1 for row in rows if "FALSE" in row[1:])
    
    print(f"\nCSV file created: {output_file}")
    print(f"\nSummary:")
    print(f"  Total images: {total_images}")
    print(f"  Images with all masks: {total_images - images_with_missing}")
    print(f"  Images with missing masks: {images_with_missing}")
    print(f"    Missing root masks: {missing_root}")
    print(f"    Missing shoot masks: {missing_shoot}")
    print(f"    Missing seed masks: {missing_seed}")

if __name__ == "__main__":
    main()