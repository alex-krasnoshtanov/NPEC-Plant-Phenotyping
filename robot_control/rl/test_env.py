"""
test_wrapper.py - Simple test that wrapper functions correctly
Run environment for 1000 steps with random actions
"""
import sys
from pathlib import Path
import numpy as np
import os

# Path setup - use script location instead of current working directory
script_dir = Path(__file__).parent  # Task 11 directory
datalabs_path = script_dir.parent  # DataLabs directory
sys.path.insert(0, str(datalabs_path))
sim_dir = datalabs_path / "simulation" / "cloned"
sys.path.insert(0, str(sim_dir))

# Change to simulation directory for URDF loading
os.chdir(sim_dir)

task11_dir = datalabs_path / "robot_control" / "rl"
sys.path.insert(0, str(task11_dir))

from ot2_env import OT2Env

def test_wrapper():
    """Run 1000 steps with random actions to verify wrapper works"""
    env = OT2Env(render=False, max_steps=300, target_threshold=0.001)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"\nRunning 1000 random steps...")
    
    total_steps = 0
    episodes = 0
    
    while total_steps < 1000:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()
            print(f"Episode {episodes} complete at step {total_steps}")
    
    env.close()
    print(f"\nTest complete: {total_steps} steps across {episodes} episodes")
    print("Wrapper functioning correctly!")

if __name__ == "__main__":
    test_wrapper()