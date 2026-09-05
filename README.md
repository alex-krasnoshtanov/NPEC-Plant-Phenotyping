# NPEC Plant Phenotyping - Autonomous Root-Tip Inoculation

An end-to-end computer-vision and robotics pipeline that locates plant root tips in
petri-dish images and drives a simulated Opentrons OT-2 pipette to dispense liquid at
each tip. Built around the NPEC (Netherlands Plant Eco-phenotyping Centre) task set.

The pipeline has four stages:

1. **Data preparation** - clean, pad and patch raw plate images/masks for training
   (`data_preparation/`).
2. **Root segmentation** - a U-Net (PyTorch / segmentation-models-pytorch) segments
   roots, followed by connected-component analysis and primary-root-length measurement
   (`segmentation/`).
3. **Robot control** - PID and reinforcement-learning (PPO) controllers move the OT-2
   pipette to target coordinates in a PyBullet simulation (`robot_control/`,
   `simulation/`).
4. **Integrated system** - the full loop: detect dish -> segment roots -> find tips ->
   transform pixel to robot coordinates -> move and dispense, with benchmarking
   (`integrated_system/`).

## Layout

```
data_preparation/     image/mask preprocessing, padding, patchify pipeline
segmentation/         U-Net training, evaluation, inference, root analysis
simulation/cloned/    shared PyBullet OT-2 environment (sim_class + URDF/meshes/textures)
robot_control/pid/    PID tuning + kinematics notebooks
robot_control/rl/     PPO env wrappers, training/eval, saved policies, results
integrated_system/    orchestrator, perception modules, benchmarking
docs/                 stage-by-stage write-ups and the full report
```

## Running

```bash
uv sync            # or: pip install -r requirements.txt
```

Notebooks and scripts locate the shared simulation automatically (they search upward
for `simulation/cloned`), so they can be run from their own folders.

## Not included in the repository

Two things are deliberately excluded (see `.gitignore`) and must be supplied locally:

- **Segmentation weights** (`best_model.pth`, ~279 MB) - exceeds the GitHub file limit.
  Train them with `segmentation/train.py`; see `integrated_system/model/README.md`.
- **NPEC plate images** (`simulation/cloned/textures/_plates/`) - third-party research
  data, not redistributed here. See that folder's README to supply your own.

Because of this, the integrated system is documented and inspectable but not runnable
end-to-end from a fresh clone without those assets.

## Credits

The reinforcement-learning environment wrapper and the group policy
(`robot_control/rl/ot2_env.py`, `models/ppo_group_1mm.zip`) were developed
collaboratively; the wrapper is credited to **Aaron Ciuffo**. The individual
exploration variant (`ot2_env_exploration.py`, `models/ppo_individual_5mm.zip`) is the
author's own. The OT-2 PyBullet simulation under `simulation/cloned/` is provided
NPEC/course scaffolding, included here so the pipeline is legible.

## Licence

[MIT](LICENSE), covering the work in this repository that is the author's own.

Two things are **not** covered by it and are not relicensed here: the OT-2 PyBullet
simulation under `simulation/cloned/`, which is NPEC/course scaffolding provided to
the project, and the NPEC *Arabidopsis* plate images, which are third-party research
data and are not redistributed at all. See Credits above.
