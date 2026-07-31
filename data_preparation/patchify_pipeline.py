"""
Simplified patch pipeline for image-root mask pairs.

Core Pipeline:
1. Find image and corresponding mask
2. Check if mask has non-zero values (skip blank masks)
3. Crop petri dish from image
4. Pad image with mirror padding (256x256, stride 128)
5. Crop mask using same coordinates as image
6. Patch both image and mask
7. Save patches in training structure

Design choices:
- Crop coordinates from image applied to mask ensures perfect alignment
- Mirror padding (BORDER_REFLECT_101) avoids artificial edges
- patchify output has shape (n_h, n_w, 1, patch_h, patch_w, channels)
  so we need [i, j, 0, :, :, :] indexing to extract patches correctly
"""

import cv2
import numpy as np
from pathlib import Path
from patchify import patchify
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class CropCoords:
    """Store crop boundaries to apply same crop to both image and mask."""
    x1: int
    y1: int
    x2: int
    y2: int


def find_mask_file(image_path: Path, mask_dir: Path, mask_type: str = "root") -> Optional[Path]:
    """
    Find corresponding mask for an image.
    
    Args:
        image_path: Path to the image file
        mask_dir: Directory containing masks
        mask_type: Type of mask to find ('root', 'shoot', or 'seed')
    
    Returns:
        Path to mask file, or None if not found
    """
    base_name = image_path.stem
    
    # Strategy 1: Direct glob search for mask type
    # This handles various naming patterns robustly
    pattern = f"{base_name}*{mask_type}_mask*"
    matches = list(mask_dir.glob(pattern))
    
    if matches:
        # If multiple matches, prefer .tif over .tiff
        tif_matches = [m for m in matches if m.suffix == '.tif']
        if tif_matches:
            return tif_matches[0]
        return matches[0]
    
    # Strategy 2: Explicit pattern matching (fallback)
    patterns = [
        f"{base_name}-Fish Eye Corrected_{mask_type}_mask.tif",
        f"{base_name}-Fish Eye Corrected_{mask_type}_mask.tiff",
        f"{base_name}_{mask_type}_mask.tif",
        f"{base_name}_{mask_type}_mask.tiff",
        f"{base_name}_{mask_type}.tif",
    ]
    
    for pattern in patterns:
        mask_path = mask_dir / pattern
        if mask_path.exists():
            return mask_path
    
    return None


def has_content(mask: np.ndarray) -> bool:
    """Check if mask has any non-zero values."""
    return np.any(mask > 0)


def detect_petri_dish(image: np.ndarray, shrink: int = 20) -> CropCoords:
    """
    Detect petri dish and return crop coordinates.

    Algorithm:
    1. Otsu threshold to separate dish from background
    2. Morphological operations to clean noise
    3. Find largest contour (the dish)
    4. Create square bounding box
    5. Shrink inward to avoid black edges
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Otsu finds optimal threshold automatically
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up: remove small noise, fill holes
    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find largest contour (petri dish)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found - cannot detect petri dish")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Shrink to avoid black edges
    x += shrink
    y += shrink
    w -= 2 * shrink
    h -= 2 * shrink

    # Make square, centered on dish
    size = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)

    return CropCoords(x1, y1, x2, y2)


def apply_crop(array: np.ndarray, coords: CropCoords) -> np.ndarray:
    """Apply crop coordinates to image or mask."""
    return array[coords.y1:coords.y2, coords.x1:coords.x2]


def calculate_padding(h: int, w: int, patch_size: int = 256, overlap: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Calculate padding to satisfy patchify requirements.

    Patchify requires: (size - patch_size) % step == 0
    We split padding symmetrically to keep crop centered.

    Returns: (top, bottom, left, right)
    """
    step = int(patch_size * (1 - overlap))

    # Calculate needed padding
    def pad_needed(size):
        if size < patch_size:
            return patch_size - size
        remainder = (size - patch_size) % step
        return (step - remainder) % step if remainder != 0 else 0

    h_pad = pad_needed(h)
    w_pad = pad_needed(w)

    # Split symmetrically
    top = h_pad // 2
    bottom = h_pad - top
    left = w_pad // 2
    right = w_pad - left

    return top, bottom, left, right


def pad_array(array: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    Pad array with mirror reflection.

    BORDER_REFLECT_101 mirrors without repeating edge pixels.
    This avoids artificial discontinuities in biological images.
    """
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return array
    return cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def create_patches(array: np.ndarray, patch_size: int = 256, overlap: float = 0.5) -> np.ndarray:
    """
    Create overlapping patches using patchify.

    Returns shape: (n_h, n_w, 1, patch_size, patch_size, channels)
    Note: middle dimension is always 1 (patchify artifact)
    """
    # Ensure channel dimension exists
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)

    step = int(patch_size * (1 - overlap))
    patches = patchify(array, (patch_size, patch_size, array.shape[2]), step=step)

    return patches


def extract_patch(patches: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Extract individual patch from patchify output.

    CRITICAL: Must use [i, j, 0, :, :, :] indexing.
    The '0' index handles patchify's singleton dimension.
    """
    patch = patches[i, j, 0, :, :, :]

    # Remove singleton channel if present (for masks)
    if patch.shape[2] == 1:
        patch = patch[:, :, 0]

    return patch


def load_and_convert_image(path: str) -> np.ndarray:
    """
    Load image and convert to RGB uint8.

    Handles: grayscale -> RGB conversion
    Assumes 3-channel images saved as identical channels are already RGB
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # Convert to uint8 if needed
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)

    # Handle color channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        # Assume BGR from OpenCV, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_and_convert_mask(path: str) -> np.ndarray:
    """Load mask and ensure grayscale uint8."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    if mask.dtype == np.uint16:
        mask = (mask / 256).astype(np.uint8)
    elif mask.dtype in [np.float32, np.float64]:
        mask = (mask * 255).astype(np.uint8)

    return mask


def process_pair(
        image_path: Path,
        mask_path: Path,
        patch_size: int = 256,
        overlap: float = 0.5,
        shrink: int = 20
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], dict]:
    """
    Process single image-mask pair through complete pipeline.

    Returns:
        List of (image_patch, mask_patch) tuples
        Metadata dictionary with processing info
    """
    # Load
    image = load_and_convert_image(str(image_path))
    mask = load_and_convert_mask(str(mask_path))

    # Check mask has content
    if not has_content(mask):
        return [], {'skipped': True, 'reason': 'blank_mask'}

    # Detect petri dish crop
    coords = detect_petri_dish(image, shrink)

    # Crop both using same coordinates
    cropped_img = apply_crop(image, coords)
    cropped_mask = apply_crop(mask, coords)

    # Ensure exact dimension match (handles edge cases)
    h = min(cropped_img.shape[0], cropped_mask.shape[0])
    w = min(cropped_img.shape[1], cropped_mask.shape[1])
    cropped_img = cropped_img[:h, :w]
    cropped_mask = cropped_mask[:h, :w]

    # Calculate and apply padding
    top, bottom, left, right = calculate_padding(h, w, patch_size, overlap)
    padded_img = pad_array(cropped_img, top, bottom, left, right)
    padded_mask = pad_array(cropped_mask, top, bottom, left, right)

    # Create patches
    img_patches = create_patches(padded_img, patch_size, overlap)
    mask_patches = create_patches(padded_mask, patch_size, overlap)

    # Extract individual patches
    n_h, n_w = img_patches.shape[0], img_patches.shape[1]
    patch_pairs = []

    for i in range(n_h):
        for j in range(n_w):
            img_patch = extract_patch(img_patches, i, j)
            mask_patch = extract_patch(mask_patches, i, j)

            # Basic validation: check dimensions match
            if img_patch.shape[:2] != mask_patch.shape:
                continue

            patch_pairs.append((img_patch, mask_patch))

    metadata = {
        'skipped': False,
        'original_shape': image.shape,
        'cropped_shape': (h, w),
        'padded_shape': padded_img.shape,
        'padding': (top, bottom, left, right),
        'num_patches': len(patch_pairs),
        'grid_size': (n_h, n_w)
    }

    return patch_pairs, metadata


def save_patches(
        patch_pairs: List[Tuple[np.ndarray, np.ndarray]],
        output_img_dir: Path,
        output_mask_dir: Path,
        base_name: str
) -> int:
    """
    Save patch pairs to disk.

    Returns: number of successfully saved patches
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for idx, (img_patch, mask_patch) in enumerate(patch_pairs):
        filename = f"{base_name}_patch_{str(idx).zfill(4)}.png"

        img_path = output_img_dir / filename
        mask_path = output_mask_dir / filename

        # Convert RGB to BGR for saving
        img_bgr = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)

        # Save both
        if cv2.imwrite(str(img_path), img_bgr) and cv2.imwrite(str(mask_path), mask_patch):
            saved += 1
        else:
            # If one fails, delete both to maintain consistency
            img_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)

    return saved


def process_dataset(
        images_dir: str,
        masks_dir: str,
        output_images_dir: str,
        output_masks_dir: str,
        patch_size: int = 256,
        overlap: float = 0.5,
        shrink: int = 20,
        mask_type: str = "root"
):
    """
    Process complete dataset of image-mask pairs.

    Scans images_dir for images, finds corresponding masks in masks_dir,
    processes valid pairs, and saves patches in training structure.
    """
    img_dir = Path(images_dir)
    mask_dir = Path(masks_dir)
    out_img_dir = Path(output_images_dir)
    out_mask_dir = Path(output_masks_dir)

    # Find all images
    image_files = sorted(
        list(img_dir.glob('*.png')) +
        list(img_dir.glob('*.jpg')) +
        list(img_dir.glob('*.jpeg')) +
        list(img_dir.glob('*.tif')) +
        list(img_dir.glob('*.tiff'))
    )

    print(f"Found {len(image_files)} images")

    # Statistics
    processed = 0
    skipped_no_mask = 0
    skipped_blank = 0
    total_patches = 0
    errors = []

    # Process each image
    for img_file in tqdm(image_files, desc="Processing"):
        # Find mask
        mask_file = find_mask_file(img_file, mask_dir, mask_type=mask_type)
        if mask_file is None:
            skipped_no_mask += 1
            continue

        try:
            # Process pair
            patch_pairs, metadata = process_pair(
                img_file, mask_file, patch_size, overlap, shrink
            )

            # Check if skipped
            if metadata.get('skipped', False):
                if metadata['reason'] == 'blank_mask':
                    skipped_blank += 1
                continue

            # Save patches
            saved = save_patches(
                patch_pairs, out_img_dir, out_mask_dir, img_file.stem
            )

            processed += 1
            total_patches += saved

        except Exception as e:
            errors.append({'file': img_file.name, 'error': str(e)})

    # Summary
    print("\n" + "=" * 60)
    print(f"Processed: {processed} images")
    print(f"Total patches: {total_patches}")
    print(f"Skipped (no mask): {skipped_no_mask}")
    print(f"Skipped (blank mask): {skipped_blank}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:3]:
            print(f"  - {err['file']}: {err['error']}")
    print("=" * 60)


if __name__ == "__main__":
    # usage for train set
    process_dataset(
        images_dir="../data/processed/dataset_unified/train_images",
        masks_dir="../data/processed/dataset_unified/train_masks",
        output_images_dir="../data/processed/dataset_unified_patches/train_images",
        output_masks_dir="../data/processed/dataset_unified_patches/train_masks",
        patch_size=256,
        overlap=0.5,
        shrink=20,
        mask_type="root"
    )

    # usage for validation set
    process_dataset(
        images_dir="../data/processed/dataset_unified/val_images",
        masks_dir="../data/processed/dataset_unified/val_masks",
        output_images_dir="../data/processed/dataset_unified_patches/val_images",
        output_masks_dir="../data/processed/dataset_unified_patches/val_masks",
        patch_size=256,
        overlap=0.5,
        shrink=20,
        mask_type="root"
    )