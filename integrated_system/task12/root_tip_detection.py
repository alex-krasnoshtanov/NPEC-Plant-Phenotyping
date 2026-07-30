"""
Root tip detection module.

This module:
1. Separates individual plants from the full root mask (instance segmentation)
2. Finds the bottommost point (root tip) of each plant
3. Returns tip coordinates in pixel space
"""

import numpy as np
import cv2
from typing import List, Tuple


def separate_plants(root_mask: np.ndarray, 
                   num_plants: int = 5,
                   plant_start: int = 350,
                   plant_step: int = 500,
                   roi_width: int = 250) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Separate individual plants using expected positions and ROI-based assignment.
    
    Strategy:
    - We know plants are arranged horizontally at roughly equal spacing
    - Define expected X positions for each plant
    - Assign connected components to nearest plant based on overlap
    
    Args:
        root_mask: Binary mask (0 or 255)
        num_plants: Number of plants expected (typically 5)
        plant_start: Starting x-position for first plant
        plant_step: Distance between plant centers
        roi_width: Half-width of region of interest around each position
        
    Returns:
        individual_roots: List of binary masks, one per plant
        labels: Label map showing plant assignments (0=background, 1-5=plants)
    """
    print(f"\n  Separating {num_plants} plants:")
    print(f"    Expected positions: start={plant_start}, step={plant_step}")
    
    # Binarize and clean
    binary_mask = (root_mask > 128).astype(np.uint8) * 255
    
    # Clean up noise with more aggressive morphological operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    # Remove small noise with opening
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
    # Fill small holes with closing
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    # Remove noise again after closing
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small)
    
    h, w = cleaned.shape
    
    # Calculate expected positions
    expected_positions = [plant_start + i * plant_step for i in range(num_plants)]
    print(f"    Expected X positions: {expected_positions}")
    
    # Find all connected components
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    
    print(f"    Found {num_labels-1} connected components")
    
    # Minimum component size to filter out noise (adjust based on expected root size)
    min_component_size = 50  # pixels
    
    # Create masks for each plant
    plant_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(num_plants)]
    
    # Assign each component to nearest plant
    for comp_id in range(1, num_labels):  # Skip background (0)
        # Filter out small components (likely noise)
        comp_area = stats[comp_id, cv2.CC_STAT_AREA]
        if comp_area < min_component_size:
            continue
        comp_mask = (labels_map == comp_id)
        
        # Get X extent of component
        x_coords = np.where(comp_mask.any(axis=0))[0]
        if len(x_coords) == 0:
            continue
        
        comp_x_min = x_coords.min()
        comp_x_max = x_coords.max()
        comp_center = (comp_x_min + comp_x_max) / 2
        
        # Find which plant ROI has best overlap
        best_plant = None
        best_overlap = 0
        
        for plant_idx, x_center in enumerate(expected_positions):
            # Define ROI bounds
            roi_x_min = max(0, x_center - roi_width)
            roi_x_max = min(w, x_center + roi_width)
            
            # Calculate overlap
            overlap_start = max(comp_x_min, roi_x_min)
            overlap_end = min(comp_x_max, roi_x_max)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_plant = plant_idx
        
        # Assign to best matching plant
        if best_plant is not None and best_overlap > 0:
            plant_masks[best_plant][comp_mask] = 255
    
    # Filter non-empty plants
    individual_roots = []
    labels_output = np.zeros_like(cleaned, dtype=np.int32)
    
    for plant_idx, plant_mask in enumerate(plant_masks):
        pixel_count = (plant_mask > 0).sum()
        if pixel_count > 0:
            print(f"    Plant {plant_idx+1}: {pixel_count:,} pixels")
            individual_roots.append(plant_mask)
            labels_output[plant_mask > 0] = plant_idx + 1
        else:
            print(f"    Plant {plant_idx+1}: NO PIXELS (skipped)")
    
    print(f"  Found {len(individual_roots)} plants with roots")
    
    return individual_roots, labels_output


def find_root_tip(root_mask: np.ndarray, max_y_percentile: float = 0.95) -> np.ndarray:
    """
    Find the root tip (bottommost point) of a single plant.
    
    Strategy:
    - Find all white pixels in the mask
    - Get the pixel with the LARGEST Y coordinate (bottom of image)
    - If multiple pixels at that Y level, use the median X
    - Constrain to avoid extreme bottom (likely noise)
    
    Args:
        root_mask: Binary mask of single plant (0 or 255)
        max_y_percentile: Maximum Y position as fraction of image height (default 0.95)
                          Prevents detecting noise at very bottom of image
        
    Returns:
        [x, y] coordinates of root tip in pixel space
        Note: Returns [x, y] format (x=horizontal, y=vertical)
        In image coordinates: y=0 is top, larger y is bottom
    """
    # Find all root pixels
    y_coords, x_coords = np.where(root_mask > 0)
    
    if len(y_coords) == 0:
        return None
    
    # Apply boundary constraint to avoid noise at extreme bottom
    h = root_mask.shape[0]
    max_allowed_y = int(h * max_y_percentile)
    
    # Filter out pixels beyond the boundary
    valid_mask = y_coords <= max_allowed_y
    y_coords = y_coords[valid_mask]
    x_coords = x_coords[valid_mask]
    
    if len(y_coords) == 0:
        # If no valid pixels, fall back to original method
        y_coords, x_coords = np.where(root_mask > 0)
    
    # Find bottommost y (largest Y = bottom in image coordinates)
    max_y = y_coords.max()
    
    # Get all x coordinates at bottommost level
    bottommost_mask = (y_coords == max_y)
    bottommost_x = x_coords[bottommost_mask]
    
    # Use median x if multiple points
    tip_x = float(np.median(bottommost_x))
    tip_y = float(max_y)
    
    return np.array([tip_x, tip_y])


def detect_root_tips(prediction: np.ndarray,
                    threshold: float = 0.5,
                    num_plants: int = 5,
                    plant_start: int = 350,
                    plant_step: int = 500,
                    roi_width: int = 250,
                    save_debug: bool = True,
                    output_path: str = "debug_root_tips.png",
                    dish_image: np.ndarray = None) -> Tuple[List[np.ndarray], dict]:
    """
    Complete pipeline: prediction -> individual plants -> root tips.
    
    Args:
        prediction: Model prediction (H, W) with values [0, 1]
        threshold: Threshold for binarization
        num_plants: Expected number of plants
        plant_start: X position of first plant
        plant_step: Spacing between plants
        roi_width: Search width around each position
        save_debug: Whether to save debug visualization
        output_path: Path for debug image
        dish_image: Original dish image for overlay (optional)
        
    Returns:
        root_tips: List of [x, y] coordinates for each detected tip
        diagnostic_info: Dictionary with intermediate results for visualization/analysis
    """
    print(f"\n{'='*70}")
    print("ROOT TIP DETECTION")
    print(f"{'='*70}")
    
    # 1. Binarize prediction
    root_mask = (prediction > threshold).astype(np.uint8) * 255
    print(f"Threshold: {threshold}")
    pixel_count = (root_mask > 0).sum()
    print(f"Binary mask: {pixel_count:,} white pixels")
    
    # Early exit if no roots detected
    if pixel_count == 0:
        print(f"\n  WARNING: No roots detected in prediction!")
        print(f"  This image may not contain any plants.")
        return [], {'root_mask': root_mask, 'plant_labels': np.zeros_like(root_mask), 
                    'individual_roots': [], 'num_plants': 0, 'threshold': threshold}
    
    # 2. Separate individual plants
    individual_roots, plant_labels = separate_plants(
        root_mask,
        num_plants=num_plants,
        plant_start=plant_start,
        plant_step=plant_step,
        roi_width=roi_width
    )
    
    # 3. Find root tip for each plant
    print(f"\n  Finding root tips:")
    root_tips = []
    
    for i, plant_mask in enumerate(individual_roots):
        tip = find_root_tip(plant_mask)
        if tip is not None:
            print(f"    Plant {i+1}: Tip at [x={tip[0]:.1f}, y={tip[1]:.1f}]")
            root_tips.append(tip)
        else:
            print(f"    Plant {i+1}: No tip found (empty mask)")
    
    print(f"\n  Total tips detected: {len(root_tips)}")
    print(f"{'='*70}\n")
    
    # Prepare diagnostic information
    diagnostic_info = {
        'root_mask': root_mask,
        'plant_labels': plant_labels,
        'individual_roots': individual_roots,
        'num_plants': len(individual_roots),
        'threshold': threshold
    }
    
    # Create comprehensive debug visualization
    if save_debug and len(root_tips) > 0:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Rectangle
            import matplotlib.patches as mpatches
            
            fig = plt.subplots(2, 3, figsize=(20, 13))
            fig, axes = plt.subplots(2, 3, figsize=(20, 13))
            
            # 1. Original prediction heatmap
            im1 = axes[0, 0].imshow(prediction, cmap='hot', vmin=0, vmax=1)
            axes[0, 0].set_title('Model Prediction (probability map)', fontsize=14)
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            
            # 2. Binary mask
            axes[0, 1].imshow(root_mask, cmap='gray')
            axes[0, 1].set_title(f'Binary Mask (threshold={threshold})', fontsize=14)
            axes[0, 1].axis('off')
            
            # 3. Plant separation (colored by plant ID)
            colored_labels = np.zeros((*plant_labels.shape, 3), dtype=np.uint8)
            colors = [
                [255, 0, 0],     # Red
                [0, 255, 0],     # Green
                [0, 0, 255],     # Blue
                [255, 255, 0],   # Yellow
                [255, 0, 255],   # Magenta
            ]
            for plant_id in range(1, len(individual_roots) + 1):
                mask = plant_labels == plant_id
                colored_labels[mask] = colors[(plant_id - 1) % len(colors)]
            
            axes[0, 2].imshow(colored_labels)
            axes[0, 2].set_title('Plant Separation (instance segmentation)', fontsize=14)
            axes[0, 2].axis('off')
            
            # Create legend for plant colors
            patches = [mpatches.Patch(color=np.array(colors[i])/255, label=f'Plant {i+1}') 
                      for i in range(min(len(individual_roots), len(colors)))]
            axes[0, 2].legend(handles=patches, loc='upper right')
            
            # 4. Overlay on dish image (if provided)
            if dish_image is not None:
                overlay = dish_image.copy()
                mask_colored = np.zeros_like(overlay)
                mask_colored[root_mask > 0] = [0, 255, 0]  # Green for roots
                overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                axes[1, 0].imshow(overlay)
                axes[1, 0].set_title('Root Mask Overlay on Original Image', fontsize=14)
            else:
                axes[1, 0].imshow(root_mask, cmap='gray')
                axes[1, 0].set_title('Binary Mask (no original image provided)', fontsize=14)
            axes[1, 0].axis('off')
            
            # 5. Root tips visualization on mask
            axes[1, 1].imshow(root_mask, cmap='gray')
            for i, tip in enumerate(root_tips):
                circle = Circle((tip[0], tip[1]), radius=15, color='red', fill=True)
                axes[1, 1].add_patch(circle)
                axes[1, 1].text(tip[0], tip[1]-30, f'{i+1}', color='red', 
                               fontsize=12, fontweight='bold', ha='center',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[1, 1].set_title(f'Detected Root Tips (n={len(root_tips)})', fontsize=14)
            axes[1, 1].axis('off')
            
            # 6. Expected plant positions overlay
            axes[1, 2].imshow(root_mask, cmap='gray', alpha=0.5)
            expected_positions = [plant_start + i * plant_step for i in range(num_plants)]
            h = root_mask.shape[0]
            for i, x_pos in enumerate(expected_positions):
                # Draw ROI
                roi_rect = Rectangle((x_pos - roi_width, 0), 2*roi_width, h, 
                                    linewidth=2, edgecolor='yellow', 
                                    facecolor='yellow', alpha=0.2)
                axes[1, 2].add_patch(roi_rect)
                # Draw center line
                axes[1, 2].axvline(x=x_pos, color='yellow', linestyle='--', linewidth=2)
                axes[1, 2].text(x_pos, 50, f'P{i+1}', color='yellow', 
                               fontsize=12, fontweight='bold', ha='center',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Overlay actual tips
            for i, tip in enumerate(root_tips):
                circle = Circle((tip[0], tip[1]), radius=15, color='red', fill=True)
                axes[1, 2].add_patch(circle)
            
            axes[1, 2].set_title('Expected Positions vs Detected Tips', fontsize=14)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Debug visualization saved to: {output_path}")
        except Exception as e:
            print(f"  Warning: Could not save debug visualization: {e}")
            import traceback
            traceback.print_exc()
    
    return root_tips, diagnostic_info


if __name__ == "__main__":
    """Test root tip detection with synthetic data."""
    print("Testing root tip detection...")
    
    # Create synthetic root mask
    h, w = 2734, 2734
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create 5 vertical "roots" at expected positions
    for i in range(5):
        x_center = 350 + i * 500
        # Draw vertical line with some width
        for y in range(500, 2500):
            width = max(3, 15 - (y - 500) // 200)  # Taper toward bottom
            mask[y, max(0, x_center-width):min(w, x_center+width)] = 255
    
    # Detect tips
    tips, diagnostic = detect_root_tips(mask / 255.0, threshold=0.5)
    
    print(f"\nTest completed: {len(tips)} tips found")
    for i, tip in enumerate(tips):
        print(f"  Tip {i+1}: {tip}")