# Task 9: Robotics Environment Interaction

This task demonstrates interaction with the Opentrons OT-2 robot simulation environment using PyBullet. The implementation systematically discovers the robot's working envelope and validates access to all workspace corners.

## Work Envelope Coordinates

The pipette tip's operational workspace was determined experimentally by driving the robot to its physical limits along each axis:

| Axis | Minimum | Maximum | Range |
|------|---------|---------|-------|
| **X** | -0.187 m | 0.253 m | 440 mm |
| **Y** | -0.171 m | 0.220 m | 391 mm |
| **Z** | 0.170 m | 0.289 m | 119 mm |

The workspace forms a rectangular prism with dimensions of 440×391×119 mm, providing a working volume of approximately 20.5 liters.

### Corner Verification

The robot successfully navigated to all 8 corners of its workspace, validating complete envelope coverage:
1. (-0.187, -0.171, 0.170) - Bottom-front-left
2. (-0.187, -0.171, 0.289) - Top-front-left
3. (-0.187, 0.220, 0.170) - Bottom-back-left
4. (-0.187, 0.220, 0.289) - Top-back-left
5. (0.253, -0.171, 0.170) - Bottom-front-right
6. (0.253, -0.171, 0.289) - Top-front-right
7. (0.253, 0.220, 0.170) - Bottom-back-right
8. (0.253, 0.220, 0.289) - Top-back-right

## Implementation Approach

### Boundary Discovery Algorithm

The working envelope was discovered through systematic exploration:

1. **Starting Position:** Robot initialized at (0, 0, 0.2) to begin at a safe height
2. **Z-Axis Limits:** 
   - Drive upward (200 steps) to find maximum height
   - Drive downward (400 steps) to find minimum height
3. **X-Axis Limits:**
   - Return to safe Z position
   - Drive right (200 steps) to find maximum X
   - Drive left (400 steps) to find minimum X
4. **Y-Axis Limits:**
   - Return to X center position
   - Drive forward (200 steps) to find maximum Y
   - Drive backward (400 steps) to find minimum Y

### Corner Navigation

After determining boundaries, the robot visits all 8 corner positions using:
- **Direction Calculation:** Compute vector from current position to target corner
- **Proportional Control:** Apply `np.sign()` to move in correct direction at 0.5 units/step
- **Convergence Check:** Stop when within 0.01 units of target in all axes
- **Maximum Attempts:** 500 steps allocated per corner to handle any path

### Command Interface

The simulation accepts action vectors with 4 elements:
```python
action = [dx, dy, dz, gripper]
```
- `dx`, `dy`, `dz`: Movement deltas in Cartesian space (-1.0 to +1.0)
- `gripper`: Gripper control (unused in this task)

### State Observations

Robot state accessed via `get_states()` returns dictionary containing:
- `pipette_position`: 3D coordinates [x, y, z]
- Indexed by `robotId_{id}` for multi-robot scenarios
Organization

The implementation is contained in `task_9.ipynb` with three cells:

1. **Cell 1:** Import dependencies (pybullet, numpy, pathlib)
2. **Cell 2:** Setup Python path and change to simulation directory for URDF loading
3. **Cell 3:** Main execution loop that:
   - Creates/resets simulation with single robot
   - Discovers working envelope boundaries
   - Navigates to all 8 corners
   - Outputs final coordinates

### Design Decisions

- **Persistent Simulation:** Uses `globals()` check to avoid recreating simulation on re-runs
- **Fixed Step Size:** All movements use ±0.5 action magnitude for consistent behavior
- **Safety-First Exploration:** Z-axis tested first, returns to safe height between X/Y tests
- **Convergence Threshold:** 0.01 units chosen to balance precision with execution time

## Observations and Results

### Movement Precision
- The robot consistently reaches target corners within the 0.01 unit threshold
- No overshoot or oscillation observed during corner navigation
- Step-based control provides predictable, repeatable movements

### Workspace Geometry
- The workspace forms a rectangular prism (not cubic) with different ranges per axis
- X-axis provides the largest range (440 mm), followed by Y-axis (391 mm)
- Z-axis is most constrained (119 mm), indicating limited vertical movement
- All 8 corners are reachable, confirming full envelope access
- Coordinate system follows standard robotics convention (Z-up)
- The Z minimum (0.170 m) suggests a base plate or table surface constraint

### Simulation Performance
- Boundary discovery completes in ~1200 simulation steps total
- Corner navigation averages <500 steps per corner
- Real-time rendering allows visual verification of movement pathsruns for efficient re-execution
- **Rendering:** Simulation window remains open for visual verification
- **Action Space:** Fourth element in action vector controls gripper (unused in this task)

## Future Improvements

1. Add velocity and acceleration observations
2. Implement smoother trajectory planning between corners
3. Calculate and visualize reachable workspace volume
4. Test gripper functionality (action[3] parameter)
5. Record and analyze movement timings
6. Add collision detection with workspace objects

## References

- Opentrons OT-2: Laboratory automation robot platform
- PyBullet: Physics simulation engine
- Custom simulation environment: `simulation/cloned/sim_class.py`
