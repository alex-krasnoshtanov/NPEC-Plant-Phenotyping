# 3. Robot Control

The OT-2 pipette is driven to target coordinates in a PyBullet simulation
(`simulation/cloned/sim_class.py`). Two controllers are provided.

## PID (`robot_control/pid/`)
- `tuning.py` - grid/manual tuning of PID gains per axis, saving `tuning_results.json`.
- `kinematics.ipynb`, `tuning.ipynb` - the exploration behind the chosen gains.

PID is deterministic and needs no training; it is the controller used by the integrated
system for the final approach-and-dispense move.

## Reinforcement learning (`robot_control/rl/`)
- `ot2_env.py` - Gymnasium wrapper around the simulation (group policy, 1 mm success
  threshold). Credited to Aaron Ciuffo.
- `ot2_env_exploration.py` - individual variant with a 5 mm threshold.
- `evaluate.py`, `test_env.py` - policy evaluation and an env smoke test.
- `models/ppo_group_1mm.zip`, `models/ppo_individual_5mm.zip` - trained PPO policies
  (Stable-Baselines3).
- `results/group/`, `results/individual/` - trajectory and convergence plots.
- `training_and_eval.ipynb` - training and rollout visualization.

RL demonstrates a learned alternative to PID; the tighter 1 mm policy is the stronger of
the two.
