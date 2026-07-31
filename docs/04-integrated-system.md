# 4. Integrated System

`integrated_system/dispensing_system.py` runs the full loop:

1. Render a plate image from the simulation.
2. `perception/petri_detection.py` - find the dish boundary.
3. `perception/model_inference.py` - patch-based U-Net inference to a root mask.
4. `perception/root_tip_detection.py` - locate root tips in the mask.
5. Transform pixel coordinates into the robot frame.
6. Drive the pipette (PID) to each tip and dispense.

`benchmarking.ipynb` evaluates the loop (including loading an RL policy from
`robot_control/rl/models/`) and writes summary figures to `statistics/`.

Running end-to-end needs the segmentation weights (`model/best_model.pth`) and plate
images (`simulation/cloned/textures/_plates/`), neither of which is committed - see the
repo README.
