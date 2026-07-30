from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path


def filter_mask_by_components(mask: np.ndarray, 
                              top_n: int = 5, 
                              min_area: int = 60) -> np.ndarray:
    """
    Pre-filter mask to remove noise by keeping only largest components.
    
    Args:
        mask: Binary mask (0 or 255)
        top_n: Maximum number of components to keep
        min_area: Minimum area in pixels for a component
    
    Returns:
        Filtered binary mask
        
    Design choice: This preprocessing step removes noise before plant detection.
    Alternative: Could skip filtering and detect all components, but this would
    include small artifacts that aren't actual roots.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    label_ids = np.arange(1, num_labels)
    
    # Filter by minimum area
    valid_mask = areas >= min_area
    valid_areas = areas[valid_mask]
    valid_labels = label_ids[valid_mask]
    
    if len(valid_labels) == 0:
        return np.zeros_like(mask)
    
    # Keep only top N largest
    actual_top_n = min(top_n, len(valid_labels))
    largest_n = valid_labels[np.argsort(valid_areas)[-actual_top_n:]]
    
    # Create filtered mask
    output = np.zeros_like(mask)
    for lbl in largest_n:
        output[labels == lbl] = 255
    
    return output


def assign_components_to_plants(mask: np.ndarray,
                               expected_positions: List[int],
                               roi_width: int = 250) -> List[dict]:
    """
    Assign connected components to plants based on ROI overlap.
    Returns full-sized masks for each plant (not cropped).
    
    Args:
        mask: Binary mask (filtered, with plants as white)
        expected_positions: List of x-coordinates where plants are expected (left to right)
        roi_width: Half-width of search region around each position
    
    Returns:
        List of exactly len(expected_positions) plant dictionaries with full-sized masks
        
    Design choice: Component-based assignment instead of ROI cropping.
    This preserves complete root structures even when they extend beyond ROI boundaries.
    Alternative: ROI cropping (previous approach) was simpler but could cut off root tips.
    """
    h, w = mask.shape
    
    # Find all connected components in the full image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Initialize plant data structures
    plants = []
    plant_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(len(expected_positions))]
    
    # For each component (skip background label 0)
    component_assignments = {}
    
    for comp_id in range(1, num_labels):
        comp_mask = (labels == comp_id)
        comp_area = stats[comp_id, cv2.CC_STAT_AREA]
        
        # Calculate overlap with each plant ROI
        max_overlap = 0
        assigned_plant = -1
        
        for plant_idx, x_center in enumerate(expected_positions):
            x_min = max(0, x_center - roi_width)
            x_max = min(w, x_center + roi_width)
            
            # Count pixels in this component that fall within this ROI
            roi_overlap = comp_mask[:, x_min:x_max].sum()
            
            if roi_overlap > max_overlap:
                max_overlap = roi_overlap
                assigned_plant = plant_idx
        
        # Assign component to plant with maximum overlap
        if assigned_plant >= 0 and max_overlap > 0:
            component_assignments[comp_id] = assigned_plant
            plant_masks[assigned_plant][comp_mask] = 255
    
    # Create plant dictionaries
    for plant_idx, x_center in enumerate(expected_positions):
        x_min = max(0, x_center - roi_width)
        x_max = min(w, x_center + roi_width)
        
        plant_mask = plant_masks[plant_idx]
        
        # Check if any roots assigned to this plant
        if plant_mask.sum() == 0:
            plants.append({
                'plant_id': plant_idx,
                'plant_label': f'Plant_{plant_idx}',
                'found': False,
                'expected_x': x_center,
                'roi_bounds': (x_min, x_max),
                'num_roots': 0,
                'total_area': 0,
                'root_info': [],
                'plant_mask': plant_mask  # Full-sized, all zeros
            })
            continue
        
        # Segment individual roots within this plant's mask
        labeled_roots, root_info = segment_roots_in_plant_mask(
            plant_mask, plant_idx
        )
        
        # Calculate plant-level statistics
        total_area = sum(r['area'] for r in root_info)
        
        plants.append({
            'plant_id': plant_idx,
            'plant_label': f'Plant_{plant_idx}',
            'found': True,
            'expected_x': x_center,
            'roi_bounds': (x_min, x_max),
            'num_roots': len(root_info),
            'total_area': total_area,
            'labeled_roots': labeled_roots,
            'root_info': root_info,
            'plant_mask': plant_mask  # Full-sized mask with only this plant's roots
        })
    
    return plants


def segment_roots_in_plant_mask(plant_mask: np.ndarray,
                                plant_id: int = 0,
                                min_root_area: int = 50) -> Tuple[np.ndarray, List[dict]]:
    """
    Segment individual roots within a plant's full-sized mask.
    
    Args:
        plant_mask: Binary mask (full image size) with only this plant's roots
        plant_id: ID of the parent plant (for clear root labeling)
        min_root_area: Minimum area for a root to be counted
    
    Returns:
        labeled_mask: Array where each root has unique integer label (same size as input)
        root_info: List of dicts containing info about each root
        
    Design choice: Works on full-sized masks to preserve spatial context.
    """
    labeled_mask, num_roots = label(plant_mask)
    
    root_info = []
    for root_id in range(1, num_roots + 1):
        root_mask = (labeled_mask == root_id)
        area = root_mask.sum()
        
        if area >= min_root_area:
            y_coords, x_coords = np.where(root_mask)
            root_info.append({
                'plant_id': plant_id,
                'root_id': root_id,
                'root_label': f'Plant_{plant_id}_Root_{root_id}',
                'area': area,
                'centroid_x': x_coords.mean(),
                'centroid_y': y_coords.mean(),
                'bbox': (x_coords.min(), x_coords.max(), 
                        y_coords.min(), y_coords.max())
            })
        else:
            # Remove small roots from labeled mask
            labeled_mask[labeled_mask == root_id] = 0
    
    return labeled_mask, root_info


def process_single_image(image_name: str,
                         mask: np.ndarray,
                         start: int = 350,
                         step: int = 500,
                         num_plants: int = 5,
                         roi_width: int = 250,
                         filter_top_n: int = 5,
                         filter_min_area: int = 100,
                         min_root_area: int = 50) -> Tuple[np.ndarray, List[dict]]:
    # Step 1: SKIP filtering - just use the mask directly
    # filtered_mask = filter_mask_by_components(mask, filter_top_n, filter_min_area)
    filtered_mask = mask  # Use original mask without filtering
    
    # Step 2: Calculate exactly num_plants positions from left to right
    expected_positions = [start + i * step for i in range(num_plants)]
    
    # Step 3: Assign components to plants and segment their roots
    plants = assign_components_to_plants(
        filtered_mask, 
        expected_positions, 
        roi_width
    )
    
    # Step 4: Segment roots within each plant
    for plant in plants:
        if plant['found']:
            labeled_roots, root_info = segment_roots_in_plant_mask(
                plant['plant_mask'],
                plant['plant_id'],
                min_root_area
            )
            plant['labeled_roots'] = labeled_roots
            plant['root_info'] = root_info
            plant['num_roots'] = len(root_info)
            plant['total_area'] = sum(r['area'] for r in root_info)
    
    return filtered_mask, plants


def visualize_plant_segmentation(image_name: str,
                                 original_mask: np.ndarray,
                                 filtered_mask: np.ndarray,
                                 plants: List[dict],
                                 image: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None):
    """
    Visualize the complete segmentation: original mask, filtered mask, 
    detected plants with individual roots color-coded.
    
    Args:
        image_name: Name of the image
        original_mask: Original binary mask
        filtered_mask: Mask after component filtering
        plants: Detection and segmentation results
        image: Optional original image to display
        save_path: Optional path to save the figure
        
    Design choice: Shows ROI boundaries and color-coded roots on full image.
    """
    num_panels = 3 if image is None else 4
    fig, axes = plt.subplots(1, num_panels, figsize=(6*num_panels, 8))
    
    panel_idx = 0
    
    # Panel 1: Original image (if provided)
    if image is not None:
        axes[panel_idx].imshow(image)
        axes[panel_idx].set_title('Original Image')
        axes[panel_idx].axis('off')
        panel_idx += 1
    
    # Panel 2: Original mask
    axes[panel_idx].imshow(original_mask, cmap='gray')
    axes[panel_idx].set_title('Original Mask')
    axes[panel_idx].axis('off')
    panel_idx += 1
    
    # Panel 3: Filtered mask
    axes[panel_idx].imshow(filtered_mask, cmap='gray')
    axes[panel_idx].set_title('Filtered Mask')
    axes[panel_idx].axis('off')
    panel_idx += 1
    
    # Panel 4: Segmented plants with ROIs
    h, w = filtered_mask.shape
    composite = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate distinct colors for each plant
    np.random.seed(42)
    
    for plant in plants:
        x_min, x_max = plant['roi_bounds']
        
        if plant['found']:
            # Draw ROI boundary (green for found plants)
            cv2.rectangle(composite, (x_min, 0), (x_max, h-1), (0, 255, 0), 2)
            
            # Color each root uniquely within this plant
            labeled_roots = plant.get('labeled_roots')
            if labeled_roots is not None:
                for root in plant['root_info']:
                    root_id = root['root_id']
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    
                    # Create mask for this specific root (on full image)
                    root_mask = (labeled_roots == root_id)
                    composite[root_mask] = color
            
            # Add plant label at top of ROI
            label_x = (x_min + x_max) // 2
            cv2.putText(composite, f'P{plant["plant_id"]}', 
                       (label_x - 20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Draw ROI boundary (red for missing plants)
            cv2.rectangle(composite, (x_min, 0), (x_max, h-1), (255, 0, 0), 2)
            
            # Add plant label for missing plants too
            label_x = (x_min + x_max) // 2
            cv2.putText(composite, f'P{plant["plant_id"]}', 
                       (label_x - 20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    axes[panel_idx].imshow(composite)
    
    # Add text annotations
    found_count = sum(p['found'] for p in plants)
    total_roots = sum(p['num_roots'] for p in plants if p['found'])
    axes[panel_idx].set_title(
        f'Root Segmentation\n{found_count}/{len(plants)} plants found, '
        f'{total_roots} total roots'
    )
    axes[panel_idx].axis('off')
    
    plt.suptitle(image_name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def create_detailed_plant_view(plant: dict, 
                               image_name: str,
                               save_path: Optional[str] = None):
    """
    Create detailed visualization of a single plant with all its roots.
    
    Args:
        plant: Plant dictionary from assign_components_to_plants()
        image_name: Name of the image (for title)
        save_path: Optional path to save figure
        
    Design choice: Shows full plant structure on original image dimensions,
    highlighting the ROI region where this plant is expected.
    """
    if not plant['found']:
        print(f"Plant {plant['plant_id']} not found, skipping detailed view")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Full plant mask with ROI highlighted
    plant_mask = plant['plant_mask']
    axes[0].imshow(plant_mask, cmap='gray')
    
    # Draw ROI boundary
    x_min, x_max = plant['roi_bounds']
    h, w = plant_mask.shape
    roi_rect = patches.Rectangle((x_min, 0), x_max - x_min, h,
                                 linewidth=2, edgecolor='g', facecolor='none')
    axes[0].add_patch(roi_rect)
    
    axes[0].set_title(f'Plant {plant["plant_id"]}: Full Mask\n(Green box = expected ROI)')
    axes[0].axis('off')
    
    # Right: Colored root segmentation
    labeled_roots = plant.get('labeled_roots')
    if labeled_roots is not None:
        h, w = labeled_roots.shape
        
        # Create colored visualization
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(plant['plant_id'])  # Consistent colors per plant
        
        for root in plant['root_info']:
            root_id = root['root_id']
            color = tuple(np.random.randint(50, 255, 3).tolist())
            mask = (labeled_roots == root_id)
            colored[mask] = color
        
        axes[1].imshow(colored)
        axes[1].set_title(
            f'{plant["num_roots"]} roots detected\n'
            f'Total area: {plant["total_area"]} pixels'
        )
        axes[1].axis('off')
        
        # Add root labels on the colored image
        for root in plant['root_info']:
            cx, cy = root['centroid_x'], root['centroid_y']
            axes[1].text(cx, cy, str(root['root_id']), 
                        color='white', fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
    
    plt.suptitle(f'{image_name} - Plant {plant["plant_id"]}', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def get_segmentation_summary(plants: List[dict]) -> dict:
    """
    Generate summary statistics for segmentation results.
    
    Args:
        plants: List of plant dictionaries
    
    Returns:
        Dictionary with summary statistics
    """
    found_plants = [p for p in plants if p['found']]
    
    if not found_plants:
        return {
            'total_expected': len(plants),
            'total_found': 0,
            'detection_rate': 0.0,
            'total_roots': 0,
            'avg_roots_per_plant': 0.0,
            'total_area': 0,
            'avg_area_per_plant': 0.0,
            'avg_area_per_root': 0.0
        }
    
    total_roots = sum(p['num_roots'] for p in found_plants)
    total_area = sum(p['total_area'] for p in found_plants)
    
    return {
        'total_expected': len(plants),
        'total_found': len(found_plants),
        'detection_rate': len(found_plants) / len(plants),
        'total_roots': total_roots,
        'avg_roots_per_plant': total_roots / len(found_plants) if found_plants else 0,
        'total_area': total_area,
        'avg_area_per_plant': total_area / len(found_plants) if found_plants else 0,
        'avg_area_per_root': total_area / total_roots if total_roots > 0 else 0
    }


def export_segmentation_data(results: Dict[str, dict], 
                             output_path: str = 'segmentation_data.txt'):
    """
    Export segmentation results in a structured text format.
    
    Args:
        results: Dictionary from process_all_images()
        output_path: Path to save the text file
        
    Format shows: Image -> Plants (left to right) -> Roots for each plant
    """
    with open(output_path, 'w') as f:
        f.write("ROOT SEGMENTATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for image_name, result in results.items():
            plants = result['plants']
            summary = result['summary']
            
            f.write(f"IMAGE: {image_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Summary: {summary['total_found']}/{summary['total_expected']} plants found, "
                   f"{summary['total_roots']} total roots\n\n")
            
            for plant in plants:
                if plant['found']:
                    f.write(f"  {plant['plant_label']} (Plant {plant['plant_id']}) - FOUND\n")
                    f.write(f"    Position: x={plant['expected_x']}, ROI: [{plant['roi_bounds'][0]}, {plant['roi_bounds'][1]}]\n")
                    f.write(f"    Roots: {plant['num_roots']}, Total area: {plant['total_area']} pixels\n")
                    
                    if plant['num_roots'] > 0:
                        f.write(f"    Root details:\n")
                        for root in plant['root_info']:
                            f.write(f"      - {root['root_label']}: area={root['area']}px, "
                                   f"centroid=({root['centroid_x']:.1f}, {root['centroid_y']:.1f})\n")
                else:
                    f.write(f"  {plant['plant_label']} (Plant {plant['plant_id']}) - NOT FOUND\n")
                    f.write(f"    Expected position: x={plant['expected_x']}, ROI: [{plant['roi_bounds'][0]}, {plant['roi_bounds'][1]}]\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Segmentation data exported to {output_path}")


def process_all_images(images: Dict[str, np.ndarray],
                      masks: Dict[str, np.ndarray],
                      start: int = 350,
                      step: int = 500,
                      num_plants: int = 5,
                      roi_width: int = 250,
                      filter_top_n: int = 5,
                      filter_min_area: int = 100,
                      min_root_area: int = 50,
                      visualize: bool = True,
                      save_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Process all images with complete segmentation pipeline.
    
    Args:
        images: Dictionary of image arrays
        masks: Dictionary of mask arrays
        start: Starting x-position for first plant
        step: Distance between plants
        num_plants: Number of expected plants per image (always creates this many)
        roi_width: Half-width of search region
        filter_top_n: Keep top N components in filtering
        filter_min_area: Minimum component area for filtering
        min_root_area: Minimum area for individual roots
        visualize: Whether to show visualizations
        save_dir: Optional directory to save results
    
    Returns:
        Dictionary mapping image names to their results
        (filtered_mask, plants, summary)
        
    Each image will have exactly num_plants plants (0 to num_plants-1),
    ordered from left to right, even if some are not found.
    """
    results = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(masks)} images...")
    print(f"Parameters: start={start}, step={step}, num_plants={num_plants}, roi_width={roi_width}")
    print(f"Filtering: top_n={filter_top_n}, min_area={filter_min_area}")
    print(f"Root segmentation: min_root_area={min_root_area}\n")
    
    for image_name in masks.keys():
        mask = masks[image_name]
        image = images.get(image_name)
        
        # Process image
        filtered_mask, plants = process_single_image(
            image_name, mask, start, step, num_plants, roi_width,
            filter_top_n, filter_min_area, min_root_area
        )
        
        # Get summary
        summary = get_segmentation_summary(plants)
        
        # Store results
        results[image_name] = {
            'filtered_mask': filtered_mask,
            'plants': plants,
            'summary': summary
        }
        
        # Print summary with plant-by-plant breakdown
        print(f"\n{image_name}:")
        print(f"  Found: {summary['total_found']}/{summary['total_expected']} plants")
        for plant in plants:
            status = "FOUND" if plant['found'] else "NOT FOUND"
            roots_info = f", {plant['num_roots']} roots" if plant['found'] else ""
            print(f"    Plant {plant['plant_id']}: {status}{roots_info}")
        print(f"  Total roots across all plants: {summary['total_roots']}")
        
        # Visualize
        if visualize:
            save_path = save_dir / f'{Path(image_name).stem}_segmentation.png' if save_dir else None
            visualize_plant_segmentation(
                image_name, mask, filtered_mask, plants, image, save_path
            )
    
    # Export text summary
    if save_dir:
        export_segmentation_data(results, save_dir / 'segmentation_data.txt')
    
    return results


def save_individual_plants(results: Dict[str, dict],
                          save_dir: str = 'individual_plants',
                          save_format: str = 'mask') -> Dict[str, List[str]]:
    """
    Save each plant as an individual file with naming: imagename_plant_N
    
    Args:
        results: Dictionary from process_all_images()
        save_dir: Directory to save individual plant files
        save_format: What to save - options:
            'mask': Binary mask of plant's roots (default)
            'colored': Colored segmentation with individual roots
            'labeled': Labeled mask with root IDs
    
    Returns:
        Dictionary mapping image names to lists of saved file paths
        
    Design choice: Saves full-sized masks (same dimensions as original image).
    This preserves complete root structures even if they extend beyond expected ROI.
    Alternative: Could crop to bounding box, but full size makes it easier to
    combine/compare plants and maintains spatial relationships.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    print(f"\nSaving individual plants to {save_dir}/")
    print(f"Format: {save_format}")
    print(f"Note: All plant masks are full-sized (same as original image)\n")
    
    for image_name, result in results.items():
        plants = result['plants']
        
        # Get base name without extension for consistent naming
        base_name = Path(image_name).stem
        
        image_files = []
        
        for plant in plants:
            # Plant numbering: 1-5 for external files (0-4 internally)
            plant_num = plant['plant_id'] + 1
            filename = f"{base_name}_plant_{plant_num}.png"
            filepath = save_dir / filename
            
            if not plant['found']:
                # Save empty image (same size as other plants)
                h, w = result['filtered_mask'].shape
                empty = np.zeros((h, w), dtype=np.uint8)
                if save_format == 'colored':
                    empty = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.imwrite(str(filepath), empty)
                
                image_files.append(str(filepath))
                print(f"  Saved {filename} (plant not found - empty full-sized image)")
                continue
            
            # Get plant mask (full-sized)
            plant_mask = plant['plant_mask']
            
            # Save based on format
            if save_format == 'mask':
                # Binary mask of all roots in this plant
                cv2.imwrite(str(filepath), plant_mask)
            
            elif save_format == 'colored':
                # Colored segmentation with each root a different color
                labeled_roots = plant.get('labeled_roots')
                if labeled_roots is not None:
                    h, w = labeled_roots.shape
                    colored = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    np.random.seed(plant['plant_id'])
                    for root in plant['root_info']:
                        root_id = root['root_id']
                        color = tuple(np.random.randint(50, 255, 3).tolist())
                        mask = (labeled_roots == root_id)
                        colored[mask] = color
                    
                    # Convert RGB to BGR for OpenCV
                    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(filepath), colored_bgr)
                else:
                    # Fallback if no labeled_roots
                    cv2.imwrite(str(filepath), plant_mask)
            
            elif save_format == 'labeled':
                # Labeled mask where each root has its ID as pixel value
                labeled_roots = plant.get('labeled_roots')
                if labeled_roots is not None:
                    # Scale labels to be visible: each root ID * 50
                    visible_labels = (labeled_roots * 50).astype(np.uint8)
                    cv2.imwrite(str(filepath), visible_labels)
                else:
                    # Fallback if no labeled_roots
                    cv2.imwrite(str(filepath), plant_mask)
            
            image_files.append(str(filepath))
            print(f"  Saved {filename} ({plant['num_roots']} roots, {plant['total_area']} pixels)")
        
        saved_files[image_name] = image_files
    
    print(f"\nTotal files saved: {sum(len(files) for files in saved_files.values())}")
    print(f"All files are full-sized masks (same dimensions as original images)")
    return saved_files


def save_individual_roots(results: Dict[str, dict],
                         save_dir: str = 'individual_roots',
                         full_size: bool = True,
                         max_spread_ratio: float = 0.4,
                         validate_connectivity: bool = True) -> Dict[str, Dict[str, List[str]]]:
    """
    Save each individual root as a separate file.
    
    Args:
        results: Dictionary from process_all_images()
        save_dir: Directory to save individual root files
        full_size: If True, save as full-sized masks (same as original image)
                   If False, save as cropped bounding boxes (legacy behavior)
        max_spread_ratio: Maximum ratio of root spread to image width (default 0.4)
                         If roots span more than this, flag as potential misassignment
        validate_connectivity: If True, check that roots within plant are spatially coherent
    
    Returns:
        Nested dictionary: {image_name: {plant_label: [root_file_paths]}}
        
    Design choice: Full-sized masks preserve spatial context and make it easy to
    combine/compare roots. Validates root connectivity to catch misassignments.
    Alternative: Cropped bounding boxes save disk space but lose spatial information.
    
    Connectivity validation approach: For each plant, calculates the horizontal spread
    of all roots. If spread exceeds max_spread_ratio, identifies outlier roots that
    are far from the median position and filters them out. This catches cases where
    roots from adjacent plants were incorrectly assigned together.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    total_roots = 0
    total_filtered = 0
    warnings = []
    
    mode_str = "full-sized masks" if full_size else "cropped bounding boxes"
    print(f"\nSaving individual roots as {mode_str} to {save_dir}/")
    if validate_connectivity:
        print(f"Validation enabled: max spread ratio = {max_spread_ratio:.1%}\n")
    
    for image_name, result in results.items():
        plants = result['plants']
        base_name = Path(image_name).stem
        
        # Get image dimensions
        h, w = result['filtered_mask'].shape
        
        image_roots = {}
        
        for plant in plants:
            if not plant['found']:
                continue
            
            plant_num = plant['plant_id'] + 1
            plant_label = f"plant_{plant_num}"
            plant_roots = []
            
            labeled_roots = plant.get('labeled_roots')
            if labeled_roots is None or plant['num_roots'] == 0:
                continue
            
            # Validate root spread within this plant
            valid_roots = plant['root_info'].copy()
            
            if validate_connectivity and len(valid_roots) > 0:
                # Calculate spread of all roots in this plant
                all_x_coords = []
                all_y_coords = []
                
                for root in valid_roots:
                    root_id = root['root_id']
                    y_coords, x_coords = np.where(labeled_roots == root_id)
                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
                
                if all_x_coords:
                    x_min_plant = min(all_x_coords)
                    x_max_plant = max(all_x_coords)
                    spread_x = x_max_plant - x_min_plant
                    spread_ratio = spread_x / w
                    
                    # Check if roots are spread too wide
                    if spread_ratio > max_spread_ratio:
                        # Try to identify outlier roots
                        root_x_positions = []
                        for root in valid_roots:
                            root_id = root['root_id']
                            y_coords, x_coords = np.where(labeled_roots == root_id)
                            root_x_center = x_coords.mean()
                            root_x_positions.append((root_id, root_x_center))
                        
                        # Calculate median x position
                        x_positions = [x for _, x in root_x_positions]
                        median_x = np.median(x_positions)
                        
                        # Filter roots that are too far from median
                        filtered_roots = []
                        outlier_roots = []
                        
                        for root in valid_roots:
                            root_id = root['root_id']
                            root_x_center = next(x for rid, x in root_x_positions if rid == root_id)
                            distance_from_median = abs(root_x_center - median_x)
                            
                            # If root is within reasonable distance from median, keep it
                            if distance_from_median < w * (max_spread_ratio / 2):
                                filtered_roots.append(root)
                            else:
                                outlier_roots.append((root_id, root_x_center))
                        
                        if outlier_roots:
                            warning_msg = (f"{image_name} - {plant_label}: "
                                         f"Filtered {len(outlier_roots)} outlier root(s) "
                                         f"(original spread: {spread_ratio:.2%}, threshold: {max_spread_ratio:.2%})")
                            warnings.append(warning_msg)
                            print(f"  WARNING: {warning_msg}")
                            total_filtered += len(outlier_roots)
                            
                            valid_roots = filtered_roots
            
            # Save each valid root
            for root in valid_roots:
                root_id = root['root_id']
                
                if full_size:
                    # Create full-sized mask with only this root
                    root_mask = np.zeros((h, w), dtype=np.uint8)
                    root_mask[labeled_roots == root_id] = 255
                    
                    filename = f"{base_name}_plant_{plant_num}_root_{root_id}.png"
                    filepath = save_dir / filename
                    cv2.imwrite(str(filepath), root_mask)
                else:
                    # Legacy: crop to bounding box
                    y_coords, x_coords = np.where(labeled_roots == root_id)
                    x_min = max(0, x_coords.min() - 10)
                    x_max = min(w, x_coords.max() + 11)
                    y_min = max(0, y_coords.min() - 10)
                    y_max = min(h, y_coords.max() + 11)
                    
                    root_crop = labeled_roots[y_min:y_max, x_min:x_max]
                    root_binary = (root_crop == root_id).astype(np.uint8) * 255
                    
                    filename = f"{base_name}_plant_{plant_num}_root_{root_id}.png"
                    filepath = save_dir / filename
                    cv2.imwrite(str(filepath), root_binary)
                
                plant_roots.append(str(filepath))
                total_roots += 1
            
            if plant_roots:
                image_roots[plant_label] = plant_roots
                status = f"saved {len(plant_roots)} roots"
                if full_size:
                    status += " (full-sized)"
                print(f"  {base_name} - {plant_label}: {status}")
        
        if image_roots:
            saved_files[image_name] = image_roots
    
    print(f"\nTotal roots saved: {total_roots}")
    if validate_connectivity and total_filtered > 0:
        print(f"Total roots filtered as outliers: {total_filtered}")
    if full_size:
        print(f"All root masks are full-sized ({h}x{w} pixels)")
    
    if warnings:
        print(f"\n{len(warnings)} plant(s) had outlier roots filtered:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return saved_files


def process_and_save_all(images: Dict[str, np.ndarray],
                        masks: Dict[str, np.ndarray],
                        output_base_dir: str = 'segmentation_output',
                        start: int = 350,
                        step: int = 500,
                        num_plants: int = 5,
                        roi_width: int = 250,
                        save_plants: bool = True,
                        save_roots: bool = False,
                        plant_format: str = 'mask',
                        roots_full_size: bool = True,
                        roots_validate: bool = True,
                        visualize: bool = True) -> Dict[str, dict]:
    """
    Complete pipeline: process images and save individual plants/roots.
    
    Args:
        images: Dictionary of image arrays
        masks: Dictionary of mask arrays
        output_base_dir: Base directory for all outputs
        start: Starting x-position for first plant
        step: Distance between plants
        num_plants: Number of expected plants per image
        roi_width: Half-width of search region
        save_plants: Whether to save individual plant files
        save_roots: Whether to save individual root files
        plant_format: Format for plant files ('mask', 'colored', 'labeled')
        roots_full_size: Whether to save roots as full-sized masks (True) or cropped (False)
        roots_validate: Whether to validate root connectivity and filter outliers
        visualize: Whether to show visualizations
    
    Returns:
        Dictionary with all results including file paths
        
    This is the main function to use for complete processing.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process all images
    print("=" * 80)
    print("STEP 1: Processing images and segmenting plants/roots")
    print("=" * 80)
    
    results = process_all_images(
        images=images,
        masks=masks,
        start=start,
        step=step,
        num_plants=num_plants,
        roi_width=roi_width,
        visualize=visualize,
        save_dir=output_base_dir / 'visualizations'
    )
    
    # Step 2: Save individual plants
    if save_plants:
        print("\n" + "=" * 80)
        print("STEP 2: Saving individual plant files")
        print("=" * 80)
        
        plant_files = save_individual_plants(
            results,
            save_dir=output_base_dir / 'individual_plants',
            save_format=plant_format
        )
        
        # Add to results
        for image_name in results.keys():
            results[image_name]['plant_files'] = plant_files.get(image_name, [])
    
    # Step 3: Save individual roots (optional)
    if save_roots:
        print("\n" + "=" * 80)
        print("STEP 3: Saving individual root files")
        print("=" * 80)
        
        root_files = save_individual_roots(
            results,
            save_dir=output_base_dir / 'individual_roots',
            full_size=roots_full_size,
            validate_connectivity=roots_validate
        )
        
        # Add to results
        for image_name in results.keys():
            results[image_name]['root_files'] = root_files.get(image_name, {})
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_base_dir}/")
    print(f"  - visualizations/     : Segmentation visualizations")
    if save_plants:
        print(f"  - individual_plants/  : Individual plant files")
    if save_roots:
        mode = "full-sized" if roots_full_size else "cropped"
        val_status = "with validation" if roots_validate else "without validation"
        print(f"  - individual_roots/   : Individual root files ({mode}, {val_status})")
    print(f"  - segmentation_data.txt : Detailed text summary")
    
    return results


if __name__ == "__main__":
    print("Root segmentation module ready (improved version).")
    print("\nKey improvements:")
    print("1. Full-sized root masks preserve spatial context")
    print("2. Connectivity validation filters out misassigned roots")
    print("3. Outlier detection based on horizontal spread")
    print("\nTo use:")
    print("1. Load your images and masks")
    print("2. Call process_and_save_all() for complete pipeline")
    print("3. Set save_roots=True and roots_full_size=True for full-sized root masks")
    print("4. Set roots_validate=True to enable connectivity validation")