# Add this diagnostic code at the start of your script, right after imports
import os
from pathlib import Path

print("="*60)
print("PATH DIAGNOSTICS")
print("="*60)
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print()

# Check if paths exist
test_paths = [
    '../data/processed/dataset_unified_patches/train_images',
    '../data/processed/dataset_unified_patches/train_masks',
    './checkpoints_custom',
]

for path_str in test_paths:
    path = Path(path_str)
    print(f"Path: {path_str}")
    print(f"  Absolute: {path.absolute()}")
    print(f"  Exists: {path.exists()}")
    if path.exists() and path.is_dir():
        png_files = list(path.glob('*.png'))
        print(f"  PNG files: {len(png_files)}")
    print()
print("="*60)
print()