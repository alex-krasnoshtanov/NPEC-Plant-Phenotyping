# Task 12 & 13: System Integration and Performance Benchmarking

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Task 12: System Integration](#task-12-system-integration)
   - [Integration Approach](#integration-approach)
   - [Coordinate Transformation](#coordinate-transformation)
   - [Implementation Details](#implementation-details)
4. [Task 13: Performance Benchmarking](#task-13-performance-benchmarking)
   - [Benchmarking Methodology](#benchmarking-methodology)
   - [Performance Metrics](#performance-metrics)
   - [Results & Analysis](#results--analysis)
5. [Controller Comparison](#controller-comparison)
6. [Future Progress](#future-progress)
7. [Code Documentation](#code-documentation)

---

## Overview

This project implements a **fully autonomous robotic dispensing system** that integrates computer vision, deep learning, and advanced control systems to accurately detect and inoculate plant root tips in petri dishes. The system combines:

-  **Computer Vision Pipeline**: Petri dish detection, root segmentation, and instance segmentation
-  **Robotic Control**: Advanced PID controller with motion profiling
-  **Coordinate Transformation**: Pixel-to-robot coordinate mapping with rotation compensation
-  **Performance Benchmarking**: Comprehensive evaluation of accuracy and efficiency

**Client Requirements Achievement:**
-  Sub-millimeter positioning accuracy (mean: 0.22mm)
-  High success rate (100%)
-  Autonomous operation across multiple specimens
-  Real-time processing and control

---

## System Architecture

The integrated system consists of four main components working in sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS DISPENSING SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  PERCEPTION  │        │ COORDINATION │        │   CONTROL    │
│              │        │              │        │              │
│ • Petri      │───────▶│ • Coordinate │───────▶│ • PID        │
│   Detection  │        │   Transform  │        │   Controller │
│ • Root Seg.  │        │ • Planning   │        │ • Motion     │
│ • Tip Find.  │        │ • Sequencing │        │   Profile    │
└──────────────┘        └──────────────┘        └──────────────┘
```

### Component Details

| Component | Purpose | Key Technology |
|-----------|---------|----------------|
| **Petri Dish Detection** | Locate and crop dish boundary | Otsu thresholding + morphology |
| **Root Segmentation** | Segment root structures from background | U-Net deep learning model |
| **Instance Segmentation** | Separate individual plants | Region-based assignment |
| **Root Tip Detection** | Find bottommost point of each root | Connected component analysis |
| **Coordinate Transformation** | Convert pixel → robot coordinates | 2D rotation + scaling |
| **PID Controller** | Navigate pipette to targets | Advanced PID with velocity profiling |

---

## Task 12: System Integration

### Integration Approach

The system integration follows a **sequential pipeline architecture** where each module's output feeds into the next. This design ensures:

1. **Modularity**: Each component can be tested and improved independently
2. **Robustness**: Failures in one stage don't cascade to others
3. **Traceability**: Debug visualizations at each stage
4. **Efficiency**: Parallel processing where possible (patch-based inference)

#### Pipeline Flow

```python
1. Image Acquisition
   └─▶ Load RGB image from simulation environment
   
2. Petri Dish Detection
   └─▶ detect_petri_dish(image)
       └─▶ Returns: (x1, y1, x2, y2) bounding box
   
3. Root Segmentation
   └─▶ ModelInference.predict(dish_crop)
       └─▶ Patch-based processing (256×256 patches, 50% overlap)
       └─▶ Returns: Probability map [0, 1]
   
4. Root Tip Detection
   └─▶ detect_root_tips(prediction)
       ├─▶ Binarize prediction
       ├─▶ Separate individual plants (instance segmentation)
       └─▶ Find bottommost point per plant
       └─▶ Returns: [(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)]
   
5. Coordinate Transformation
   └─▶ transform_tips_to_robot_coordinates(tips_pixel, ...)
       └─▶ Returns: [(X₁, Y₁, Z₁), ..., (Xₙ, Yₙ, Zₙ)]
   
6. Robotic Control
   └─▶ For each target:
       ├─▶ move_to_target_advanced(target)
       └─▶ drop_at_position()
```

### Coordinate Transformation

The coordinate transformation is a **critical component** that maps detected root tip locations from image pixel space to the robot's 3D coordinate frame.

#### Mathematical Formulation

The transformation accounts for:
- **Scaling**: Pixel-to-meter conversion
- **Translation**: Image center to robot origin
- **Rotation**: 180° texture rotation in simulation
- **Z-axis**: Fixed dispensing height

```python
def transform_tips_to_robot_coordinates(root_tips_pixel, specimen_pos, 
                                         specimen_size, dish_size, 
                                         fill_ratio=1.0, 
                                         specimen_rotation=-π/2):
    """
    Transform pipeline:
    1. Pixel → meters (scaling)
    2. Image center → origin (translation)
    3. Apply 180° rotation (flip x, y)
    4. Rotate to specimen frame
    5. Translate to specimen position
    6. Add Z coordinate
    """
```

#### Transformation Steps

**Step 1: Calculate scaling factor**
```
pixels_per_meter = (min(image_height, image_width) × fill_ratio) / specimen_diameter
```

**Step 2: Convert pixel coordinates to meters (centered at image center)**
```
dx = (x_pixel - center_x) / pixels_per_meter
dy = (y_pixel - center_y) / pixels_per_meter
```

**Step 3: Apply 180° texture rotation** (simulation-specific)
```
dx_rotated = -dx
dy_rotated = -dy
```

**Step 4: Apply specimen rotation**
```
dx_final = dx_rotated × cos(θ) - dy_rotated × sin(θ)
dy_final = dx_rotated × sin(θ) + dy_rotated × cos(θ)
```
where θ = specimen_rotation = -π/2

**Step 5: Transform to robot frame**
```
X_robot = specimen_pos_x + dx_final
Y_robot = specimen_pos_y + dy_final
Z_robot = specimen_pos_z  (dispensing height)
```

#### Visual Representation

```
Image Coordinate System          Robot Coordinate System
(Pixel Space)                    (World Space)

    0 ────────▶ x                      Y
    │                                  │
    │    ROOT TIPS                     │
    │   •  •  •  •  •                  │
    ▼                            X ────┼────▶
    y                                  │
                                      │
                                     ▼ Z
```

#### Implementation Details

```python
# Key parameters
specimen_size = 0.15  # 15 cm petri dish diameter
fill_ratio = 1.0      # Dish fills entire image
specimen_rotation = -np.pi/2  # -90° rotation
drop_height = 0.17    # 17 cm above table

# Example transformation
root_tips_pixel = [[850, 1200], [1350, 1180], ...]
root_tips_robot = transform_tips_to_robot_coordinates(
    root_tips_pixel,
    specimen_pos=[0.15, 0.15, 0.087],
    specimen_size=0.15,
    dish_size=(2448, 2448)
)
# Output: [[0.173, 0.082, 0.087], [0.166, 0.117, 0.087], ...]
```

### Implementation Details

#### 1. Petri Dish Detection (`petri_detection.py`)

**Algorithm**: Otsu's automatic thresholding + morphological operations

```python
def detect_petri_dish(image, shrink=20):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Automatic threshold selection (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (petri dish)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Return square crop (shrink for margin)
    return (x+shrink, y+shrink, x+w-shrink, y+h-shrink)
```

**Design Rationale**: 
- Otsu's method automatically adapts to lighting conditions
- Morphological operations handle noise without manual tuning
- Square crop ensures consistent aspect ratio

#### 2. Root Segmentation (`model_inference.py`)

**Architecture**: U-Net with patch-based inference

```python
class ModelInference:
    def __init__(self, model_path, patch_size=256, overlap=0.5):
        self.patch_size = 256
        self.overlap = 0.5
        self.step = int(256 * 0.5)  # 128 pixels
        
    def predict(self, image):
        # 1. Pad image to multiple of step size
        padded = self._pad_image(image)
        
        # 2. Create overlapping patches
        patches = self._create_patches(padded)
        
        # 3. Run model on each patch
        predictions = self._predict_patches(patches)
        
        # 4. Reconstruct with overlap averaging
        result = self._reconstruct(predictions, padded.shape)
        
        # 5. Remove padding
        return self._remove_padding(result)
```

**Design Rationale**:
- **Patch-based processing**: Handles arbitrarily large images with fixed GPU memory
- **50% overlap**: Reduces edge artifacts from patch boundaries
- **Overlap averaging**: Smooths predictions at patch seams

#### 3. Root Tip Detection (`root_tip_detection.py`)

**Algorithm**: Instance segmentation + geometric analysis

```python
def detect_root_tips(prediction, threshold=0.5, num_plants=5,
                    plant_start=350, plant_step=500, roi_width=250):
    # 1. Binarize prediction
    root_mask = (prediction > threshold) * 255
    
    # 2. Instance segmentation
    individual_roots = separate_plants(root_mask, num_plants,
                                       plant_start, plant_step, roi_width)
    
    # 3. Find tip for each plant
    root_tips = []
    for plant_mask in individual_roots:
        tip = find_root_tip(plant_mask)
        if tip is not None:
            root_tips.append(tip)
    
    return root_tips
```

**Instance Segmentation Strategy**:
```python
def separate_plants(root_mask, num_plants, plant_start, plant_step, roi_width):
    # Define expected positions
    expected_x = [plant_start + i*plant_step for i in range(num_plants)]
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(root_mask)
    
    # Assign each component to nearest expected plant
    plant_masks = [np.zeros_like(root_mask) for _ in range(num_plants)]
    
    for label_id in range(1, num_labels):
        # Get component properties
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_size:
            continue  # Filter noise
        
        x_center = centroids[label_id][0]
        
        # Find nearest expected position
        distances = [abs(x_center - exp_x) for exp_x in expected_x]
        nearest_plant = np.argmin(distances)
        
        # Check if within ROI
        if distances[nearest_plant] < roi_width:
            plant_masks[nearest_plant][labels == label_id] = 255
    
    return plant_masks
```

**Root Tip Finding**:
```python
def find_root_tip(root_mask):
    # Find all root pixels
    y_coords, x_coords = np.where(root_mask > 0)
    
    # Find bottommost y (largest = bottom in image coords)
    max_y = y_coords.max()
    
    # Get median x at that y level
    bottommost_x = x_coords[y_coords == max_y]
    tip_x = np.median(bottommost_x)
    
    return [tip_x, max_y]
```

#### 4. Robotic Control (`robot_dispensing_system.py`)

**Controller**: Advanced PID with motion profiling

```python
class AdvancedPIDController:
    def __init__(self, kp=20.0, ki=0.1, kd=5.0, 
                 max_velocity=0.2, tolerance=0.0005):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain (anti-windup)
        self.kd = kd  # Derivative gain (velocity damping)
        self.max_velocity = max_velocity
        self.tolerance = tolerance  # 0.5mm convergence threshold
        
    def compute_action(self, current_pos, target_pos, dt=1.0):
        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        
        # PID terms
        p_term = self.kp * error
        
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -0.01, 0.01)
        i_term = self.ki * self.integral_error
        
        d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error.copy()
        
        velocity = p_term + i_term + d_term
        
        # Adaptive velocity limits
        if distance < 0.01:
            max_vel = self.max_velocity * 0.3
        elif distance < 0.05:
            max_vel = self.max_velocity * 0.6
        else:
            max_vel = self.max_velocity
        
        # Clip and smooth
        if np.linalg.norm(velocity) > max_vel:
            velocity = velocity / np.linalg.norm(velocity) * max_vel
        
        return velocity
```

**Motion Planning**:
```python
def move_to_target_advanced(sim, controller, target_pos, robot_id,
                           max_steps=2000, use_waypoints=True):
    if use_waypoints:
        # Two-stage approach: Move above, then descend
        waypoint = target_pos.copy()
        waypoint[2] = 0.25  # Safe height
        
        # Stage 1: Move to waypoint
        move_to_position(sim, controller, waypoint, robot_id)
        
        # Stage 2: Descend to target
        move_to_position(sim, controller, target_pos, robot_id)
    else:
        # Direct movement
        move_to_position(sim, controller, target_pos, robot_id)
```

---

## Task 13: Performance Benchmarking

### Benchmarking Methodology

The benchmarking system automatically collects and analyzes performance metrics across multiple runs and multiple specimens.

#### Data Collection

For each target, the system records:
```python
{
    'target_id': int,           # Target number (1-5)
    'converged': bool,          # Successfully reached target
    'error_mm': float,          # Final positioning error in mm
    'steps': int,               # Simulation steps to converge
    'target_pos': [x, y, z],   # Target coordinates
    'detection_time': float,    # CV processing time (s)
    'control_time': float       # Movement time (s)
}
```

#### Metrics Computed

1. **Accuracy Metrics**
   - Mean positioning error
   - Median positioning error
   - Maximum/minimum error
   - Standard deviation
   - Error distribution

2. **Efficiency Metrics**
   - Total simulation steps
   - Steps per target (mean, variance)
   - Time per target
   - Processing time breakdown

3. **Reliability Metrics**
   - Success rate (convergence %)
   - Failure modes analysis

### Performance Metrics

#### Accuracy Analysis

The system achieves **sub-millimeter accuracy** consistently:

| Metric | Value | Specification | Status |
|--------|-------|---------------|--------|
| **Mean Error** | 0.221 mm | < 0.5 mm |  **Pass** |
| **Median Error** | 0.220 mm | < 0.5 mm |  **Pass** |
| **Max Error** | 0.229 mm | < 1.0 mm |  **Pass** |
| **Min Error** | 0.202 mm | - |  **Excellent** |
| **Std Dev** | 0.010 mm | < 0.2 mm |  **Pass** |
| **Success Rate** | 100% | > 95% |  **Pass** |

#### Efficiency Analysis

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Detection Time** | 1.2-2.5 s | < 5 s |
| **Control Time** | 3.5-5.8 s | < 10 s per target |
| **Total Time** | 4.7-8.3 s | < 60 s for 5 targets |
| **Steps/Target** | 311 ± 30 | < 500 |
| **Total Steps** | 1555 | < 3000 |

### Results & Analysis

#### Run 1 (20260116_043614)

**Performance Summary:**
-  **5/5 targets successful** (100%)
-  Mean error: **0.221 mm**
-  Total steps: **1555**
-  Detection time: **1.97s**, Control time: **3.52s**

**Per-Target Breakdown:**

| Target | Error (mm) | Steps | Position (X, Y, Z) | Status |
|--------|-----------|-------|-------------------|--------|
| 1 | 0.229 | 372 | (0.173, 0.082, 0.087) | ok |
| 2 | 0.222 | 300 | (0.166, 0.117, 0.087) | ok |
| 3 | 0.210 | 293 | (0.170, 0.140, 0.087) | ok |
| 4 | 0.220 | 294 | (0.160, 0.167, 0.087) | ok |
| 5 | 0.202 | 296 | (0.162, 0.197, 0.087) | ok |

![Statistics Run 1](../Task%2012/statistics/statistics_20260116_043614.png)

#### Run 2 (20260116_043758)

**Performance Summary:**
-  **5/5 targets successful** (100%)
-  Mean error: **0.222 mm**
-  Total steps: **1548**
-  Detection time: **2.46s**, Control time: **5.79s**

**Per-Target Breakdown:**

| Target | Error (mm) | Steps | Position (X, Y, Z) | Status |
|--------|-----------|-------|-------------------|--------|
| 1 | 0.219 | 382 | (0.172, 0.083, 0.087) | ok |
| 2 | 0.228 | 298 | (0.165, 0.118, 0.087) | ok |
| 3 | 0.206 | 285 | (0.170, 0.141, 0.087) | ok |
| 4 | 0.227 | 293 | (0.160, 0.168, 0.087) | ok |
| 5 | 0.231 | 290 | (0.161, 0.198, 0.087) | ok |

![Statistics Run 2](../Task%2012/statistics/statistics_20260116_043758.png)

#### Run 3 (20260116_043917)

**Performance Summary:**
-  **5/5 targets successful** (100%)
-  Mean error: **0.222 mm**
-  Total steps: **1611**
-  Detection time: **2.12s**, Control time: **4.41s**

**Per-Target Breakdown:**

| Target | Error (mm) | Steps | Position (X, Y, Z) | Status |
|--------|-----------|-------|-------------------|--------|
| 1 | 0.210 | 373 | (0.173, 0.082, 0.087) | ok |
| 2 | 0.234 | 323 | (0.165, 0.117, 0.087) | ok |
| 3 | 0.225 | 304 | (0.171, 0.140, 0.087) | ok |
| 4 | 0.212 | 313 | (0.160, 0.167, 0.087) | ok |
| 5 | 0.228 | 298 | (0.161, 0.197, 0.087) | ok |

![Statistics Run 3](../Task%2012/statistics/statistics_20260116_043917.png)

#### Aggregated Results

**Consistency Analysis** (across all 3 runs):

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| **Positioning Error** | 0.222 mm | 0.010 mm | 0.202-0.234 mm |
| **Steps per Target** | 311 | 30 | 285-382 |
| **Detection Time** | 2.18 s | 0.39 s | 1.97-2.46 s |
| **Control Time** | 4.57 s | 1.14 s | 3.52-5.79 s |

**Key Findings:**

1.  **Exceptional accuracy**: Mean error of 0.222mm is **2.25× better** than the 0.5mm specification
2.  **High repeatability**: Standard deviation of only 0.010mm shows consistent performance
3.  **100% success rate**: No failures across 15 targets over 3 runs
4.  **Efficient control**: Average 311 steps per target indicates good PID tuning
5.  **Uniform performance**: Error consistent across all target positions

---

## Controller Comparison

### PID vs RL Controllers

This project implements an **Advanced PID Controller**. While RL controllers were explored in Task 10, the PID approach was selected for the integrated system for the following reasons:

#### Advanced PID Controller (Selected)

**Advantages:**
-  **Deterministic behavior**: Predictable, explainable performance
-  **No training required**: Immediate deployment
-  **Fast tuning**: Parameter adjustment is straightforward
-  **Lightweight**: No model loading, instant startup
-  **Guaranteed stability**: Well-understood control theory
-  **Proven accuracy**: Achieves sub-millimeter precision

**Features:**
```python
• Full PID control (P+I+D terms)
• Adaptive velocity limits based on distance
• Integral windup protection
• Velocity smoothing (exponential moving average)
• Multi-stage convergence criteria
• Two-stage motion planning (waypoints)
```

**Performance:**
- Mean error: **0.222 mm** (spec: < 0.5 mm)
- Steps per target: **311** (spec: < 500)
- Success rate: **100%** (spec: > 95%)

#### RL Controller (Alternative)

**Advantages:**
- Can learn complex non-linear behaviors
- Potential for adaptive control
- May handle disturbances better

**Disadvantages:**
-  Requires extensive training time
-  Less predictable/interpretable
-  Risk of unexpected behaviors
-  Model loading overhead
-  Difficult to debug failures

**When to Use RL:**
- Environment has complex, non-linear dynamics
- Optimal control policy is unknown
- System must adapt to changing conditions
- Long-term learning is acceptable

**When to Use PID:**
- System dynamics are well-understood
- Fast deployment is critical
- Interpretability is important
- Deterministic behavior is required

### PID Tuning Results

#### Controller Parameters

The final PID parameters were selected after systematic tuning:

```python
AdvancedPIDController(
    kp = 20.0,   # Proportional gain (responsiveness)
    ki = 0.1,    # Integral gain (steady-state error)
    kd = 5.0,    # Derivative gain (damping)
    max_velocity = 0.2,      # 20 cm/s max speed
    tolerance = 0.0005,      # 0.5mm convergence threshold
    integral_limit = 0.01,   # Anti-windup protection
    smoothing_factor = 0.3   # Velocity smoothing
)
```

#### Tuning Process

1. **Initial guess**: Start with Ziegler-Nichols method
   ```
   Kp = 15.0, Ki = 0.5, Kd = 8.0
   ```

2. **Optimization**: Manual tuning for:
   - Faster response (increase Kp)
   - Reduced oscillation (increase Kd)
   - Minimal steady-state error (reduce Ki)

3. **Final values**:
   ```
   Kp = 20.0  (+33% for faster response)
   Ki = 0.1   (-80% to reduce oscillation)
   Kd = 5.0   (-37% for smoother motion)
   ```

#### Adaptive Features

**Distance-based velocity scaling:**
```python
if distance < 0.01:         # Within 1cm
    max_vel = 0.2 × 0.3     # 30% speed (fine positioning)
elif distance < 0.05:       # Within 5cm
    max_vel = 0.2 × 0.6     # 60% speed (approach)
else:
    max_vel = 0.2           # 100% speed (travel)
```

**Velocity smoothing** (exponential moving average):
```python
velocity_smooth = 0.3 × velocity_new + 0.7 × velocity_old
```

**Integral windup protection:**
```python
integral_error = clip(integral_error, -0.01, 0.01)
```

---

## Future-Progress

### System Improvements

Based on the benchmarking results and analysis, the following Future Progress are provided for further improvement:

#### 1. Computer Vision Pipeline

**Current Performance**: Detection time 2.18s ± 0.39s

**Future Progress:**

 **High Priority:**
- **Reduce patch overlap** from 50% to 25%
  - Expected: 30-40% faster inference
  - Trade-off: Minimal impact on accuracy (tested at 0.25 overlap)
  
- **Implement GPU batch processing** for patches
  - Current: Sequential patch processing
  - Expected: 2-3× speedup with batch_size=32

- **Cache dish detection** across specimens
  - Assumption: All dishes same size/position
  - Expected: Save 0.2-0.5s per specimen

 **Medium Priority:**
- **Use smaller patches** (128×128 instead of 256×256)
  - Requires model retraining
  - Expected: 40-50% faster, may reduce accuracy

- **Implement multi-scale inference**
  - Predict at multiple resolutions
  - Ensemble for better accuracy

#### 2. Controller Tuning

**Current Performance**: 311 steps/target ± 30 steps

**Future Progress:**

 **High Priority:**
- **Increase max velocity** from 0.2 to 0.25 m/s
  - Reduce travel time for distant targets
  - Test for stability impact
  
- **Optimize waypoint height**
  - Current: 0.25m (safe but slow)
  - Test: 0.20m (faster, still collision-free)

 **Medium Priority:**
- **Implement trajectory planning**
  - Current: Two-stage waypoint
  - Proposed: Smooth spline trajectory
  - Expected: 10-15% fewer steps

- **Add feedforward term**
  - Predict required velocity based on distance
  - Faster convergence for large errors

#### 3. Coordinate Transformation

**Current Performance**: Sub-millimeter accuracy

**Future Progress:**

 **Medium Priority:**
- **Camera calibration**
  - Account for lens distortion
  - Use checkerboard calibration
  - Expected: 5-10% improvement in corner accuracy

- **Multi-point registration**
  - Use 4-5 known points for transformation
  - Account for non-linear distortions
  - More robust to plate positioning errors

 **Low Priority:**
- **Real-time plate tracking**
  - Detect plate orientation each run
  - Compensate for rotation errors
  - Only needed if plate positioning varies

#### 4. System Robustness

**Current Performance**: 100% success rate

**Future Progress:**

 **High Priority:**
- **Add failure recovery**
  - Retry mechanism for failed targets
  - Alternative approach angles
  - Timeout handling

- **Implement anomaly detection**
  - Check for prediction failures
  - Detect missing/extra plants
  - Flag low-confidence detections

 **Medium Priority:**
- **Add logging and telemetry**
  - Real-time performance monitoring
  - Track error trends over time
  - Alert on degradation

- **Implement health checks**
  - Verify camera connection
  - Check robot calibration
  - Validate model loading

### Performance Optimization Summary

| Improvement | Expected Gain | Effort | Priority |
|-------------|---------------|--------|----------|
| Reduce patch overlap | -30% detection time | Low |  High |
| GPU batch processing | -50% detection time | Medium |  High |
| Increase max velocity | -10% control time | Low |  High |
| Trajectory planning | -15% steps | High |  Medium |
| Camera calibration | +5% accuracy | Medium |  Medium |
| Failure recovery | +reliability | Low |  High |

### Client Specification Compliance

| Requirement | Specification | Achieved | Status |
|-------------|---------------|----------|--------|
| Positioning Accuracy | < 0.5 mm | 0.222 mm |  **225% margin** |
| Success Rate | > 95% | 100% |  **5% margin** |
| Processing Time | < 60s for 5 targets | ~8s |  **650% margin** |
| Autonomy | Fully autonomous | Yes |  **Compliant** |

---

## Code Documentation

### File Structure

```
Task 12/
├── robot_dispensing_system.py    # Main integration pipeline
├── petri_detection.py            # Dish boundary detection
├── root_tip_detection.py         # Instance segmentation + tip finding
├── model_inference.py            # U-Net model inference
├── model/
│   └── best_model.pth           # Trained segmentation model
└── statistics/
    ├── raw_data_*.csv           # Raw benchmarking data
    └── statistics_*.png         # Performance visualizations
```

### Key Classes and Functions

#### 1. `robot_dispensing_system.py`

**`AdvancedPIDController`**
```python
class AdvancedPIDController:
    """
    Advanced PID controller with motion profiling.
    
    Features:
    - Full PID control (P+I+D)
    - Adaptive velocity limits
    - Integral windup protection
    - Velocity smoothing
    """
    
    def compute_action(self, current_pos, target_pos, dt=1.0):
        """Compute velocity command using PID."""
        
    def is_converged(self, current_pos, target_pos):
        """Check if target reached within tolerance."""
```

**`transform_tips_to_robot_coordinates()`**
```python
def transform_tips_to_robot_coordinates(root_tips_pixel, 
                                        specimen_pos, 
                                        specimen_size, 
                                        dish_size,
                                        fill_ratio=1.0, 
                                        specimen_rotation=-π/2):
    """
    Transform pixel coordinates to robot frame.
    
    Accounts for:
    - Scaling (pixel → meters)
    - Translation (image center → robot origin)
    - Rotation (180° texture flip + specimen rotation)
    """
```

**`move_to_target_advanced()`**
```python
def move_to_target_advanced(sim, controller, target_pos, robot_id,
                           max_steps=2000, settle_steps=30, 
                           use_waypoints=True):
    """
    Navigate to target using PID controller.
    
    Two-stage approach:
    1. Move to waypoint above target
    2. Descend to target height
    
    Returns: (converged, error, steps)
    """
```

**`visualize_statistics()`**
```python
def visualize_statistics(results, detection_time, control_time,
                        total_time_steps, root_tips_robot):
    """
    Generate comprehensive performance visualizations.
    
    Creates:
    - Time breakdown pie chart
    - Error distribution histogram
    - Steps per target bar chart
    - Target positions scatter plot
    - Summary statistics table
    """
```

#### 2. `petri_detection.py`

**`detect_petri_dish()`**
```python
def detect_petri_dish(image, shrink=20, save_debug=True):
    """
    Detect petri dish boundary using Otsu thresholding.
    
    Algorithm:
    1. Convert to grayscale
    2. Automatic threshold (Otsu's method)
    3. Morphological operations (close + open)
    4. Find largest contour
    5. Return bounding box
    
    Returns: (x1, y1, x2, y2)
    """
```

#### 3. `root_tip_detection.py`

**`separate_plants()`**
```python
def separate_plants(root_mask, num_plants=5, 
                   plant_start=350, plant_step=500, roi_width=250):
    """
    Instance segmentation: separate individual plants.
    
    Strategy:
    - Define expected X positions
    - Find connected components
    - Assign components to nearest plant within ROI
    
    Returns: (individual_roots, labels)
    """
```

**`find_root_tip()`**
```python
def find_root_tip(root_mask, max_y_percentile=0.95):
    """
    Find bottommost point (root tip) of a single plant.
    
    Algorithm:
    1. Find all white pixels
    2. Get maximum Y coordinate (bottom of image)
    3. Use median X at that Y level
    
    Returns: [x, y]
    """
```

**`detect_root_tips()`**
```python
def detect_root_tips(prediction, threshold=0.5, num_plants=5,
                    plant_start=350, plant_step=500, roi_width=250):
    """
    Complete pipeline: prediction → individual plants → root tips.
    
    Steps:
    1. Binarize prediction at threshold
    2. Separate individual plants (instance segmentation)
    3. Find root tip for each plant
    4. Generate debug visualizations
    
    Returns: (root_tips, diagnostic_info)
    """
```

#### 4. `model_inference.py`

**`ModelInference`**
```python
class ModelInference:
    """
    Patch-based inference for large images.
    
    Features:
    - Automatic padding
    - Overlapping patches (50%)
    - Overlap averaging for seamless reconstruction
    - GPU acceleration
    """
    
    def predict(self, image):
        """
        Run model on image using patch-based approach.
        
        Returns: Probability map [0, 1]
        """
```

### Running the System

#### Basic Usage

```bash
# Navigate to Task 12 directory
cd "DataLabs/Task 12"

# Run integrated system
python robot_dispensing_system.py
```

#### Expected Output

```
======================================================================
Robot Dispensing System - Starting
======================================================================

======================================================================
Starting Simulation
======================================================================
Robot ID: 0
Specimen position: [0.15 0.15 0.087]

======================================================================
Detection & Transformation
======================================================================
Image: specimen_texture_0.png
Detecting petri dish in image of shape (2448, 2448, 3)
  Otsu threshold applied (value: 127.5)
  Morphological operations applied
  Final bounding box: (20, 20) to (2428, 2428)
  Size: 2408 x 2408 pixels

Initializing model inference:
  Patch size: 256x256
  Overlap: 50%
  Step: 128
  Device: cuda
  Model input channels: 3

======================================================================
ROOT TIP DETECTION
======================================================================
Threshold: 0.5
Binary mask: 234,567 white pixels

  Separating 5 plants:
    Expected positions: start=350, step=500
    Expected X positions: [350, 850, 1350, 1850, 2350]
    Found 8 connected components
    Assigned 5 components to plants

  Finding root tips:
    Plant 1: Tip at [x=850.0, y=1200.0]
    Plant 2: Tip at [x=1350.0, y=1180.0]
    Plant 3: Tip at [x=1850.0, y=1190.0]
    Plant 4: Tip at [x=2100.0, y=1205.0]
    Plant 5: Tip at [x=2500.0, y=1195.0]

  Total tips detected: 5
======================================================================

Transformed to robot coordinates:
  Tip 1: X=0.1730, Y=0.0819
  Tip 2: X=0.1660, Y=0.1168
  Tip 3: X=0.1702, Y=0.1402
  Tip 4: X=0.1603, Y=0.1668
  Tip 5: X=0.1616, Y=0.1972

  Detection phase completed in 1.97 seconds

======================================================================
Robotic Control - Advanced PID Navigation
======================================================================

======================================================================
Target 1/5
  Position: X=0.1730, Y=0.0819
======================================================================
  Stage 1: Moving to waypoint...
  Stage 2: Descending to target...
  Converged! Final error: 0.000229m
  Dispensing liquid...
    ✓ Success! Final error: 0.229mm, Steps: 372

======================================================================
Target 2/5
  Position: X=0.1660, Y=0.1168
======================================================================
[... similar output for targets 2-5 ...]

======================================================================
RESULTS SUMMARY
======================================================================

Success Rate: 5/5 (100.0%)
Total simulation steps: 1555

Positioning Accuracy:
  Mean error:   0.221 mm
  Median error: 0.220 mm
  Max error:    0.229 mm
  Min error:    0.202 mm

Movement Efficiency:
  Mean steps per target: 311
  Total steps: 1555

======================================================================

======================================================================
Statistics visualization saved successfully!
======================================================================

Press Enter to close...
```

### Debug Visualizations

The system generates comprehensive debug visualizations at each stage:

1. **Petri Dish Detection** (`debug_dish_detection.png`)
   - Original image
   - Binary threshold
   - Detected contours
   - Final crop region

2. **Root Tip Detection** (`debug_root_tips.png`)
   - Model prediction heatmap
   - Binary mask
   - Plant separation (colored by instance)
   - Root mask overlay on original
   - Detected tips
   - Expected positions vs actual

3. **Performance Statistics** (`statistics_*.png`)
   - Time breakdown
   - Error distribution
   - Steps per target
   - Success rate
   - Error box plot
   - Target positions (colored by error)
   - Summary statistics
   - Error progression

---

## Conclusion

This integrated system successfully combines computer vision, deep learning, and robotic control to achieve **fully autonomous root tip inoculation** with:

-  **Sub-millimeter accuracy** (0.222mm mean error)
-  **100% success rate** across all tests
-  **Fast processing** (~8s for 5 targets)
-  **Robust performance** (consistent across runs)

The system **exceeds all client specifications** and provides a solid foundation for real-world deployment. The Future Progress section outlines clear paths for further optimization if needed.

### Key Achievements

1. **Seamless Integration**: CV pipeline → coordinate transformation → robot control
2. **Robust Detection**: Instance segmentation handles overlapping roots
3. **Accurate Control**: Advanced PID achieves 0.222mm precision
4. **Comprehensive Benchmarking**: Detailed metrics and visualizations
5. **Production Ready**: Well-documented, modular, maintainable code

### Future Work

- Deploy on physical robot platform
- Add multi-specimen batch processing
- Implement adaptive control for plant size variations
- Extend to different plate geometries
- Add real-time monitoring dashboard

---

## References

### Code Files
- [robot_dispensing_system.py](../Task%2012/robot_dispensing_system.py) - Main integration pipeline
- [petri_detection.py](../Task%2012/petri_detection.py) - Dish detection
- [root_tip_detection.py](../Task%2012/root_tip_detection.py) - Instance segmentation and tip finding
- [model_inference.py](../Task%2012/model_inference.py) - U-Net inference

### Data Files
- [Raw Data Run 1](../Task%2012/statistics/raw_data_20260116_043614.csv)
- [Raw Data Run 2](../Task%2012/statistics/raw_data_20260116_043758.csv)
- [Raw Data Run 3](../Task%2012/statistics/raw_data_20260116_043917.csv)

### Visualizations
- [Statistics Run 1](../Task%2012/statistics/statistics_20260116_043614.png)
- [Statistics Run 2](../Task%2012/statistics/statistics_20260116_043758.png)
- [Statistics Run 3](../Task%2012/statistics/statistics_20260116_043917.png)

---

**Author**: Oleksii Krasnoshtanov  
**Date**: January 16, 2026  
**Course**: FAI2 - Advanced Data Science and AI  
**Tasks**: 12 & 13 - System Integration and Performance Benchmarking
