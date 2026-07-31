"""
Petri dish detection module.

Uses Otsu thresholding and morphological operations to find the petri dish
boundary in the image and return a square bounding box.
"""

import numpy as np
import cv2
from typing import Tuple


def detect_petri_dish(image: np.ndarray, shrink: int = 20, save_debug: bool = True, 
                     output_path: str = "debug_dish_detection.png") -> Tuple[int, int, int, int]:
    """
    Detect petri dish boundaries using Otsu thresholding.
    
    Args:
        image: RGB or grayscale image
        shrink: Pixels to shrink from detected boundary (margin)
        save_debug: Whether to save debug visualization
        output_path: Path for debug image
        
    Returns:
        (x1, y1, x2, y2): Bounding box coordinates for cropping
        
    Design choice: Uses Otsu for automatic thresholding (no manual tuning).
    Morphological operations clean up noise. Falls back to center region if detection fails.
    
    Alternative approaches:
    - Hough circle detection (assumes perfect circle, may be too strict)
    - Edge detection + fitting (more complex)
    - Deep learning segmentation (overkill for this task)
    """
    print(f"Detecting petri dish in image of shape {image.shape}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Otsu thresholding for automatic threshold selection
    threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  Otsu threshold applied (value: {threshold_value:.1f})")
    
    # Morphological operations to clean up
    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    print(f"  Morphological operations applied")
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fallback_used = False
    if not contours:
        print(f"  WARNING: No contours found, using center 80% of image")
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.1)
        x1, y1, x2, y2 = margin, margin, w - margin, h - margin
        fallback_used = True
    else:
        # Find largest contour (should be the dish)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        print(f"  Largest contour found: area={cv2.contourArea(largest):.0f}")
        
        # Check if detected region is reasonable
        img_area = image.shape[0] * image.shape[1]
        detected_area = w * h
        
        if detected_area < 0.2 * img_area:
            print(f"  WARNING: Detected area too small ({100*detected_area/img_area:.1f}% of image)")
            print(f"           Using center 80% of image instead")
            h_img, w_img = image.shape[:2]
            margin = int(min(h_img, w_img) * 0.1)
            x1, y1, x2, y2 = margin, margin, w_img - margin, h_img - margin
            fallback_used = True
        else:
            # Shrink boundary (remove edge artifacts)
            x += shrink
            y += shrink
            w -= 2 * shrink
            h -= 2 * shrink
            
            # Make square (use maximum dimension, centered)
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(image.shape[1], x1 + size)
            y2 = min(image.shape[0], y1 + size)
    
    print(f"  Final bounding box: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Size: {x2-x1} x {y2-y1} pixels")
    
    # Create debug visualization
    if save_debug:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # Original image
            axes[0, 0].imshow(image if len(image.shape) == 3 else image, cmap='gray' if len(image.shape) == 2 else None)
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            axes[0, 0].add_patch(rect)
            axes[0, 0].set_title('Original Image with Detected Boundary', fontsize=14)
            axes[0, 0].axis('off')
            
            # Binary mask
            axes[0, 1].imshow(binary, cmap='gray')
            axes[0, 1].set_title(f'Binary Mask (Otsu threshold: {threshold_value:.1f})', fontsize=14)
            axes[0, 1].axis('off')
            
            # Cropped region
            crop = image[y1:y2, x1:x2]
            axes[1, 0].imshow(crop if len(crop.shape) == 3 else crop, cmap='gray' if len(crop.shape) == 2 else None)
            axes[1, 0].set_title(f'Cropped Dish Region ({x2-x1}x{y2-y1} px)', fontsize=14)
            axes[1, 0].axis('off')
            
            # Info text
            axes[1, 1].axis('off')
            info_text = f"""Detection Summary:
            
Image size: {image.shape[1]} x {image.shape[0]} px
Bounding box: ({x1}, {y1}) to ({x2}, {y2})
Crop size: {x2-x1} x {y2-y1} px
Shrink applied: {shrink} px
Fallback used: {fallback_used}

Detection method: Otsu thresholding
Otsu threshold value: {threshold_value:.1f}
Morphological kernel: 20x20

Crop center in original image:
  X: {(x1+x2)/2:.1f}
  Y: {(y1+y2)/2:.1f}
"""
            axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                          family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Debug visualization saved to: {output_path}")
        except Exception as e:
            print(f"  Warning: Could not save debug visualization: {e}")
    
    return x1, y1, x2, y2


if __name__ == "__main__":
    """Test petri dish detection."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python petri_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        sys.exit(1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x1, y1, x2, y2 = detect_petri_dish(img_rgb)
    
    # Visualize
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img_rgb)
    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title('Detection')
    ax1.axis('off')
    
    crop = img_rgb[y1:y2, x1:x2]
    ax2.imshow(crop)
    ax2.set_title('Cropped Dish')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('dish_detection_test.png', dpi=150)
    print(f"Saved test result to: dish_detection_test.png")