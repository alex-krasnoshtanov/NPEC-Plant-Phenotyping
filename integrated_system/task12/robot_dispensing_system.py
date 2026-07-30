"""
Robot Dispensing System for Root Tip Targeting.

This module integrates computer vision and robotic control to automatically
dispense liquid at detected root tip positions in a petri dish.

The system:
1. Captures images from the simulation environment
2. Detects petri dish boundaries and root tips using CV models
3. Transforms pixel coordinates to robot coordinate frame
4. Controls the pipette to move to targets and dispense liquid
"""

import sys
from pathlib import Path
import numpy as np
import time
import cv2
import os
import pybullet as p
import traceback
import matplotlib.pyplot as plt
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "simulation" / "cloned"))
sys.path.insert(0, str(project_root))

from petri_detection import detect_petri_dish
from model_inference import ModelInference
from root_tip_detection import detect_root_tips


def load_image_rgb(image_path):
    """Load image as RGB."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def transform_tips_to_robot_coordinates(root_tips_pixel, specimen_pos, specimen_size, 
                                          dish_size, fill_ratio=1.0, specimen_rotation=-np.pi/2):
    """
    Transform root tip pixel coordinates to robot coordinate frame.
    
    Accounts for 180° texture rotation in simulation: (x, y) → (-x, -y)
    
    Args:
        root_tips_pixel: List of [x, y] coordinates in pixel space
        specimen_pos: Robot coordinates of specimen center [x, y, z]
        specimen_size: Physical diameter of petri dish in meters
        dish_size: Image dimensions (height, width) in pixels
        fill_ratio: Fraction of image filled by dish (default 1.0)
        specimen_rotation: Rotation of specimen in robot frame (default -π/2)
        
    Returns:
        Array of robot coordinates (N, 3) for each root tip
    """
    dish_diameter_pixels = min(dish_size) * fill_ratio
    pixels_per_meter = dish_diameter_pixels / specimen_size
    
    img_height, img_width = dish_size
    center_x = img_width / 2
    center_y = img_height / 2
    
    cos_theta = np.cos(specimen_rotation)
    sin_theta = np.sin(specimen_rotation)
    
    robot_tips = []
    
    for tip_pixel in root_tips_pixel:
        # Center-origin and flip Y (image coords)
        x_from_center = tip_pixel[0] - center_x
        y_from_center = center_y - tip_pixel[1]
        
        # Scale to meters
        norm_x = x_from_center / pixels_per_meter
        norm_y = y_from_center / pixels_per_meter
        
        # Correct for 180° texture rotation
        norm_x = -norm_x
        norm_y = -norm_y
        
        # Apply specimen rotation
        rotated_x = cos_theta * norm_x - sin_theta * norm_y
        rotated_y = sin_theta * norm_x + cos_theta * norm_y
        
        # Translate to robot frame
        robot_x = specimen_pos[0] + rotated_x
        robot_y = specimen_pos[1] + rotated_y
        robot_z = specimen_pos[2]
        
        robot_tips.append([robot_x, robot_y, robot_z])
    
    return np.array(robot_tips)


class AdvancedPIDController:
    """
    Advanced PID controller with motion profiling and velocity smoothing.
    
    Features:
    - Full PID control (Proportional + Integral + Derivative)
    - Adaptive velocity limits based on distance to target
    - Integral windup protection
    - Velocity smoothing with exponential moving average
    - Multi-stage convergence criteria
    """
    
    def __init__(self, 
                 kp=15.0, ki=0.5, kd=8.0,
                 max_velocity=0.15,
                 min_velocity=0.001,
                 tolerance=0.0001,
                 fine_tolerance=0.00005,
                 integral_limit=0.01,
                 smoothing_factor=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.tolerance = tolerance
        self.fine_tolerance = fine_tolerance
        self.integral_limit = integral_limit
        self.smoothing_factor = smoothing_factor
        
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.velocity_smooth = np.zeros(3)
        self.convergence_count = 0
        self.fine_convergence_count = 0
    
    def reset(self):
        """Reset controller state."""
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.velocity_smooth = np.zeros(3)
        self.convergence_count = 0
        self.fine_convergence_count = 0
    
    def compute_action(self, current_pos, target_pos, dt=1.0):
        """Compute control action using PID with velocity limiting."""
        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral_error
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error.copy()
        
        # Combined PID output (velocity command)
        velocity = p_term + i_term + d_term
        
        # Adaptive velocity limit based on distance
        if distance < 0.01:
            current_max_velocity = self.max_velocity * 0.3
        elif distance < 0.05:
            current_max_velocity = self.max_velocity * 0.6
        else:
            current_max_velocity = self.max_velocity
        
        # Smooth velocity
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > current_max_velocity:
            velocity = velocity / velocity_magnitude * current_max_velocity
        
        # Apply exponential moving average for smoothing
        self.velocity_smooth = (self.smoothing_factor * velocity + 
                               (1 - self.smoothing_factor) * self.velocity_smooth)
        
        # Minimum velocity threshold
        if np.linalg.norm(self.velocity_smooth) < self.min_velocity:
            self.velocity_smooth = np.zeros(3)
        
        return self.velocity_smooth
    
    def is_converged(self, current_pos, target_pos, consecutive_required=5):
        """Check if controller has converged."""
        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        
        if distance < self.tolerance:
            self.convergence_count += 1
            if self.convergence_count >= consecutive_required:
                return True
        else:
            self.convergence_count = 0
        
        return False
    
    def is_fine_converged(self, current_pos, target_pos):
        """Check for fine convergence."""
        error = target_pos - current_pos
        return np.linalg.norm(error) < self.fine_tolerance


def move_to_target_advanced(sim, controller, target_pos, robot_id, 
                            max_steps=2000, settle_steps=30, use_waypoints=True):
    """
    Advanced motion control with waypoint navigation and settling time.
    
    Args:
        sim: Simulation instance
        controller: Advanced PID controller
        target_pos: Final target position [x, y, z]
        robot_id: Robot ID
        max_steps: Maximum steps per movement phase
        settle_steps: Steps to wait at target for settling
        use_waypoints: Use safe approach trajectory (recommended)
        
    Returns:
        (converged, final_error, movement_time)
    """
    states = sim.get_states()
    start_pos = np.array(states[f'robotId_{robot_id}']['pipette_position'])
    
    # Define waypoints for safe approach
    waypoints = []
    
    if use_waypoints:
        # Safe height waypoint
        safe_height = max(start_pos[2], target_pos[2]) + 0.05
        waypoint_high = target_pos.copy()
        waypoint_high[2] = safe_height
        waypoints = [waypoint_high, target_pos]
    else:
        waypoints = [target_pos]
    
    total_steps = 0
    overall_converged = True
    
    # Navigate through waypoints
    for wp_idx, waypoint in enumerate(waypoints):
        controller.reset()
        converged = False
        
        for step in range(max_steps):
            states = sim.get_states()
            current_pos = np.array(states[f'robotId_{robot_id}']['pipette_position'])
            
            # Display current pipette position
            print(f"\r    Step {step:4d}: X={current_pos[0]:7.4f}, Y={current_pos[1]:7.4f}, Z={current_pos[2]:7.4f}", end='', flush=True)
            
            # Compute control action
            velocity = controller.compute_action(current_pos, waypoint)
            
            # Execute action
            action = list(velocity) + [0]
            sim.run([action], num_steps=1)
            total_steps += 1
            
            # Check convergence
            if controller.is_converged(current_pos, waypoint):
                converged = True
                break
        
        if not converged:
            overall_converged = False
            break
    
    # Settling phase at final target
    if overall_converged:
        settle_action = [0.0, 0.0, 0.0, 0]
        sim.run([settle_action], num_steps=settle_steps)
        total_steps += settle_steps
    
    # Get final position and error
    states = sim.get_states()
    final_pos = np.array(states[f'robotId_{robot_id}']['pipette_position'])
    final_error = np.linalg.norm(target_pos - final_pos)
    
    if not overall_converged:
        print(f"    Warning: Did not converge within {max_steps} steps")
    
    return overall_converged, final_error, total_steps


def drop_at_position(sim, settle_steps=50):
    """
    Drop liquid with proper settling time.
    
    Args:
        sim: Simulation instance
        settle_steps: Steps to wait for liquid to settle (default 50)
    """
    # Trigger drop
    drop_action = [0.0, 0.0, 0.0, 1]
    sim.run([drop_action], num_steps=1)
    
    # Wait for liquid physics to settle
    settle_action = [0.0, 0.0, 0.0, 0]
    sim.run([settle_action], num_steps=settle_steps)
    print(f"    Liquid dispensed")


def visualize_statistics(results, detection_time, control_time, total_time_steps, 
                         root_tips_robot, output_dir="statistics"):
    """
    Generate and save comprehensive statistics and visualizations.
    
    Args:
        results: List of (converged, error, steps) tuples for each target
        detection_time: Time spent on detection phase (seconds)
        control_time: Time spent on control phase (seconds)
        total_time_steps: Total simulation steps
        root_tips_robot: Array of target positions
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract statistics
    success = [r[0] for r in results]
    errors_mm = [r[1] * 1000 for r in results]  # Convert to mm
    steps = [r[2] for r in results]
    
    success_count = sum(success)
    success_rate = success_count / len(results) if results else 0
    
    # Calculate timing statistics
    total_processing_time = detection_time + control_time
    avg_time_per_target = control_time / len(results) if results else 0
    
    print(f"\nGenerating statistics visualizations...")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall Time Breakdown (Pie Chart)
    ax1 = plt.subplot(3, 3, 1)
    time_data = [detection_time, control_time]
    time_labels = ['Detection\n& CV', 'Robot\nControl']
    colors_pie = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax1.pie(time_data, labels=time_labels, autopct='%1.1f%%',
                                         startangle=90, colors=colors_pie, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title(f'Time Breakdown\nTotal: {total_processing_time:.2f}s', fontsize=12, fontweight='bold')
    
    # 2. Positioning Error Distribution (Histogram)
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(errors_mm, bins=15, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(errors_mm), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_mm):.3f} mm')
    ax2.axvline(np.median(errors_mm), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors_mm):.3f} mm')
    ax2.set_xlabel('Error (mm)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Positioning Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Steps per Target (Bar Chart)
    ax3 = plt.subplot(3, 3, 3)
    target_indices = range(1, len(steps) + 1)
    bars = ax3.bar(target_indices, steps, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax3.axhline(np.mean(steps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.0f}')
    ax3.set_xlabel('Target Number', fontsize=10)
    ax3.set_ylabel('Simulation Steps', fontsize=10)
    ax3.set_title('Steps per Target', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Success Rate (Gauge-style)
    ax4 = plt.subplot(3, 3, 4)
    ax4.barh(['Success Rate'], [success_rate * 100], color='#2ca02c' if success_rate > 0.8 else '#ff9999', 
             edgecolor='black', linewidth=2)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel('Percentage (%)', fontsize=10)
    ax4.set_title(f'Success Rate: {success_count}/{len(results)} targets', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    for i, v in enumerate([success_rate * 100]):
        ax4.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    # 5. Error Box Plot
    ax5 = plt.subplot(3, 3, 5)
    box = ax5.boxplot(errors_mm, vert=True, patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('#ff7f0e')
    box['boxes'][0].set_alpha(0.7)
    ax5.set_ylabel('Error (mm)', fontsize=10)
    ax5.set_title('Error Statistics', fontsize=12, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)
    ax5.set_xticklabels(['All Targets'])
    
    # 6. Time per Target (Bar Chart)
    ax6 = plt.subplot(3, 3, 6)
    time_per_target = [s * 0.001 for s in steps]  # Assuming ~1ms per step
    bars = ax6.bar(target_indices, time_per_target, color='#9467bd', alpha=0.7, edgecolor='black')
    ax6.axhline(np.mean(time_per_target), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(time_per_target):.3f}s')
    ax6.set_xlabel('Target Number', fontsize=10)
    ax6.set_ylabel('Estimated Time (s)', fontsize=10)
    ax6.set_title('Estimated Time per Target', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, axis='y', alpha=0.3)
    
    # 7. Target Positions (2D Scatter)
    ax7 = plt.subplot(3, 3, 7)
    if len(root_tips_robot) > 0:
        x_coords = [tip[0] for tip in root_tips_robot]
        y_coords = [tip[1] for tip in root_tips_robot]
        scatter = ax7.scatter(x_coords, y_coords, c=errors_mm, cmap='RdYlGn_r', 
                             s=200, alpha=0.7, edgecolors='black', linewidths=2)
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax7.text(x, y, str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Robot X (m)', fontsize=10)
        ax7.set_ylabel('Robot Y (m)', fontsize=10)
        ax7.set_title('Target Positions (colored by error)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_aspect('equal')
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Error (mm)', fontsize=9)
    
    # 8. Summary Statistics (Text)
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    summary_text = f"""PERFORMANCE SUMMARY
    
Processing Time:
  • Detection & CV:  {detection_time:.2f}s
  • Robot Control:   {control_time:.2f}s
  • Total Time:      {total_processing_time:.2f}s
  • Avg per target:  {avg_time_per_target:.2f}s

Accuracy Metrics:
  • Mean Error:      {np.mean(errors_mm):.3f} mm
  • Median Error:    {np.median(errors_mm):.3f} mm
  • Max Error:       {np.max(errors_mm):.3f} mm
  • Min Error:       {np.min(errors_mm):.3f} mm
  • Std Dev:         {np.std(errors_mm):.3f} mm

Efficiency:
  • Total Steps:     {total_time_steps:,}
  • Mean Steps:      {np.mean(steps):.0f}
  • Total Targets:   {len(results)}
  • Success Rate:    {success_rate*100:.1f}%
"""
    ax8.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 9. Error vs Target Position (Line Plot)
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(target_indices, errors_mm, marker='o', linestyle='-', color='#d62728', 
             linewidth=2, markersize=8, alpha=0.7)
    ax9.axhline(np.mean(errors_mm), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors_mm):.3f} mm')
    ax9.set_xlabel('Target Number', fontsize=10)
    ax9.set_ylabel('Error (mm)', fontsize=10)
    ax9.set_title('Error Progression', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Robot Dispensing System - Performance Analysis\n{timestamp}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = output_path / f"statistics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comprehensive statistics to: {output_file}")
    
    # Also save a simple CSV with raw data
    csv_file = output_path / f"raw_data_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Target,Converged,Error_mm,Steps,X_pos,Y_pos,Z_pos\n")
        for i, (result, tip) in enumerate(zip(results, root_tips_robot)):
            converged, error, step_count = result
            f.write(f"{i+1},{int(converged)},{error*1000:.4f},{step_count},{tip[0]:.6f},{tip[1]:.6f},{tip[2]:.6f}\n")
    
    print(f"  Saved raw data to: {csv_file}")
    
    return str(output_file)


def main():
    """
    Main execution pipeline for automated root tip dispensing.
    
    Integrates CV detection, coordinate transformation, and robot control
    to dispense liquid at detected root tip locations.
    """
    
    print("="*70)
    print("Robot Dispensing System - Starting")
    print("="*70)
    
    sim_dir = project_root / "simulation" / "cloned"
    model_path = project_root / "Task 12" / "model" / "best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Start simulation
    print(f"\n{'='*70}")
    print("Starting Simulation")
    print(f"{'='*70}")
    
    try:
        import os
        original_dir = Path.cwd()
        os.chdir(sim_dir)
        
        from simulation.cloned.sim_class import Simulation
        import pybullet as p
        
        sim = Simulation(num_agents=1, render=True)
        robot_id = sim.robotIds[0]
        specimen_id = sim.specimenIds[0]
        
        specimen_pos_tuple, _ = p.getBasePositionAndOrientation(specimen_id)
        specimen_pos = np.array(specimen_pos_tuple)
        
        print(f"Robot ID: {robot_id}")
        print(f"Specimen position: {specimen_pos}")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Detection
    print(f"\n{'='*70}")
    print("Detection & Transformation")
    print(f"{'='*70}")
    
    # Start timing detection phase
    detection_start_time = time.time()
    
    try:
        texture_path_rel = sim.get_plate_image()
        image_path = sim_dir / texture_path_rel
        
        print(f"Image: {image_path.name}")
        
        image_rgb = load_image_rgb(image_path)
        x1, y1, x2, y2 = detect_petri_dish(image_rgb, shrink=20, save_debug=False)
        dish_crop = image_rgb[y1:y2, x1:x2]
        dish_size = (y2 - y1, x2 - x1)
        
        print(f"Dish: {dish_size[1]}x{dish_size[0]} pixels")
        
        model_inference = ModelInference(str(model_path), patch_size=256, overlap=0.5)
        prediction = model_inference.predict(dish_crop)
        
        root_tips_pixel, diagnostic_info = detect_root_tips(
            prediction, threshold=0.5, num_plants=5,
            plant_start=350, plant_step=500, roi_width=250,
            save_debug=False,
            dish_image=dish_crop
        )
        
        if len(root_tips_pixel) == 0:
            print("\n" + "="*70)
            print("NO ROOT TIPS DETECTED")
            print("="*70)
            print("This image appears to have no plants or roots.")
            print("Skipping dispensing operation.")
            print("="*70)
            
            input("\nPress Enter to close...")
            sim.close()
            os.chdir(original_dir)
            return
        
        print(f"Detected {len(root_tips_pixel)} tips")
        
        # Transform detected tips to robot coordinates
        root_tips_robot = transform_tips_to_robot_coordinates(
            root_tips_pixel,
            specimen_pos,
            0.15,  # specimen_size
            dish_size
            # Uses defaults: fill_ratio=1.0, specimen_rotation=-np.pi/2
        )
        
        print(f"Transformed to robot coordinates:")
        for i, tip in enumerate(root_tips_robot):
            print(f"  Tip {i+1}: X={tip[0]:.4f}, Y={tip[1]:.4f}")
        
        # End timing detection phase
        detection_end_time = time.time()
        detection_time = detection_end_time - detection_start_time
        print(f"\n  Detection phase completed in {detection_time:.2f} seconds")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sim.close()
        os.chdir(original_dir)
        return
    
    # Control
    print(f"\n{'='*70}")
    print("Robotic Control - Advanced PID Navigation")
    print(f"{'='*70}")
    
    # Start timing control phase
    control_start_time = time.time()
    
    # Initialize simplified PID controller for faster performance
    controller = AdvancedPIDController(
        kp=20.0,
        ki=0.1,
        kd=5.0,
        max_velocity=0.2,
        tolerance=0.0005
    )
    
    drop_height = 0.17  # Height for dispensing (17cm)
    
    results = []
    total_time_steps = 0
    
    for i, tip_pos in enumerate(root_tips_robot):
        print(f"\n{'='*70}")
        print(f"Target {i+1}/{len(root_tips_robot)}")
        print(f"  Position: X={tip_pos[0]:.4f}, Y={tip_pos[1]:.4f}")
        print(f"{'='*70}")
        
        target = tip_pos.copy()
        target[2] = drop_height
        
        try:
            # Navigate to target using advanced controller
            converged, error, move_steps = move_to_target_advanced(
                sim, controller, target, robot_id,
                max_steps=2000,
                settle_steps=30,
                use_waypoints=True
            )
            
            total_time_steps += move_steps
            
            if converged:
                # Dispense liquid
                drop_at_position(sim, settle_steps=50)
                print(f"    ✓ Success! Final error: {error*1000:.3f}mm, Steps: {move_steps}")
                results.append((True, error, move_steps))
            else:
                print(f"    ✗ Failed to converge, error: {error*1000:.3f}mm")
                results.append((False, error, move_steps))
            
            # Brief pause before next target
            time.sleep(0.2)
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((False, 1.0, 0))
    
    # End timing control phase
    control_end_time = time.time()
    control_time = control_end_time - control_start_time
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    success = sum(1 for r in results if r[0])
    errors = [r[1] * 1000 for r in results if r[0]]
    steps = [r[2] for r in results]
    
    print(f"\nSuccess Rate: {success}/{len(results)} ({success/len(results)*100:.1f}%)")
    print(f"Total simulation steps: {total_time_steps}")
    
    if errors:
        print(f"\nPositioning Accuracy:")
        print(f"  Mean error:   {np.mean(errors):.3f} mm")
        print(f"  Median error: {np.median(errors):.3f} mm")
        print(f"  Max error:    {np.max(errors):.3f} mm")
        print(f"  Min error:    {np.min(errors):.3f} mm")
    
    if steps:
        print(f"\nMovement Efficiency:")
        print(f"  Mean steps per target: {np.mean(steps):.0f}")
        print(f"  Total steps: {sum(steps)}")
    
    print(f"\n{'='*70}")
    
    # Generate and save statistics visualizations
    try:
        viz_path = visualize_statistics(
            results=results,
            detection_time=detection_time,
            control_time=control_time,
            total_time_steps=total_time_steps,
            root_tips_robot=root_tips_robot,
            output_dir="statistics"
        )
        print(f"\n{'='*70}")
        print(f"Statistics visualization saved successfully!")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\nWarning: Failed to generate statistics visualization: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to close...")
    sim.close()
    os.chdir(original_dir)


if __name__ == "__main__":
    main()
