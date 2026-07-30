"""
Test script for Alex's OT-2 Robot Model
Model: 260107.1230_alex_lr3e-4_b64_s2048_th5mm.zip

This script runs inference on the trained model and saves:
- Trajectory plots for each episode
- Distance convergence plot
- Summary statistics (JSON and text)
"""

import sys
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set up paths
script_dir = Path(__file__).parent
datalabs_path = script_dir.parent
sys.path.insert(0, str(datalabs_path))

# Add simulation/cloned to path
sim_dir = datalabs_path / "simulation" / "cloned"
sys.path.insert(0, str(sim_dir))

# Import simulation class
from simulation.cloned.sim_class import Simulation

# Change to simulation directory
original_dir = Path.cwd()
os.chdir(sim_dir)

# Add Task 11 directory to path
task11_dir = datalabs_path / "Task 11"
sys.path.insert(0, str(task11_dir))

# Import wrapper
from alex_ot2_wrapper import OT2Env

print("="*80)
print("Testing Alex's Model: 260107.1230_alex_lr3e-4_b64_s2048_th5mm")
print("="*80)

# Load the model
model_filename = "260107.1230_alex_lr3e-4_b64_s2048_th5mm.zip"
model_path = task11_dir / model_filename

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = PPO.load(str(model_path))
print(f"✓ Successfully loaded model: {model_filename}")

# Create output directory for results
output_dir = task11_dir / "test_results_alex"
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# Create environment (5mm threshold to match training)
env = OT2Env(render=False, max_steps=300, target_threshold=0.005)
print(f"✓ Environment created")
print(f"  Target threshold: {env.target_threshold * 1000}mm")
print(f"  Max steps: {env.max_steps}")

# Run inference
num_episodes = 10  # More episodes for better statistics
print(f"\nRunning {num_episodes} episodes...")

all_trajectories = []
all_goals = []
all_distances = []
all_rewards = []
all_steps = []
all_success = []

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    positions_x = []
    positions_y = []
    positions_z = []
    distances = []
    
    print(f"\nEpisode {episode + 1}/{num_episodes}")
    print(f"  Goal: X={env.goal_position[0]:.4f}m, Y={env.goal_position[1]:.4f}m, Z={env.goal_position[2]:.4f}m")
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_pos = info['current_position']
        positions_x.append(current_pos[0])
        positions_y.append(current_pos[1])
        positions_z.append(current_pos[2])
        distances.append(info['distance_to_goal'])
        
        episode_reward += reward
        steps += 1
        done = terminated or truncated
    
    # Store episode data
    all_trajectories.append({
        'x': positions_x,
        'y': positions_y,
        'z': positions_z
    })
    all_goals.append(env.goal_position.copy().tolist())
    all_distances.append(distances)
    all_rewards.append(episode_reward)
    all_steps.append(steps)
    
    final_distance_mm = info['distance_to_goal'] * 1000
    success = final_distance_mm < 5.0  # 5mm threshold
    all_success.append(success)
    
    print(f"  Steps: {steps} | Reward: {episode_reward:.2f} | Final distance: {final_distance_mm:.2f}mm | Success: {'YES' if success else 'NO'}")

print("\n" + "="*80)
print("Inference complete. Generating visualizations...")

# Generate trajectory plots for each episode
for episode_idx in range(num_episodes):
    trajectory = all_trajectories[episode_idx]
    goal = all_goals[episode_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # X-axis trajectory
    ax1.plot(trajectory['x'], linewidth=2, label='Robot Position', color='#2E86AB')
    ax1.axhline(y=goal[0], color='#E63946', linestyle='--', linewidth=2, label='Target')
    ax1.set_ylabel('X Position (m)', fontsize=12)
    ax1.set_title(f'Episode {episode_idx + 1}: X-Axis Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Y-axis trajectory
    ax2.plot(trajectory['y'], linewidth=2, label='Robot Position', color='#2E86AB')
    ax2.axhline(y=goal[1], color='#E63946', linestyle='--', linewidth=2, label='Target')
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title(f'Episode {episode_idx + 1}: Y-Axis Trajectory', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Z-axis trajectory
    ax3.plot(trajectory['z'], linewidth=2, label='Robot Position', color='#2E86AB')
    ax3.axhline(y=goal[2], color='#E63946', linestyle='--', linewidth=2, label='Target')
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Z Position (m)', fontsize=12)
    ax3.set_title(f'Episode {episode_idx + 1}: Z-Axis Trajectory', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'trajectory_episode_{episode_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {num_episodes} trajectory plots")

# Generate distance convergence plot
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
for episode_idx in range(num_episodes):
    distances_mm = [d * 1000 for d in all_distances[episode_idx]]
    success_marker = '✓' if all_success[episode_idx] else '✗'
    ax.plot(distances_mm, linewidth=2, label=f'Ep {episode_idx + 1} {success_marker}', 
            alpha=0.7, color=colors[episode_idx])

ax.axhline(y=5.0, color='#E63946', linestyle='--', linewidth=2, label='Success Threshold (5mm)')
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Distance to Goal (mm)', fontsize=12)
ax.set_title('Distance to Goal Over Time - All Episodes\nModel: 260107.1230_alex_lr3e-4_b64_s2048_th5mm', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(output_dir / 'distance_convergence_all_episodes.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved distance convergence plot")

# Generate summary statistics plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Success rate
success_count = sum(all_success)
ax1.bar(['Success', 'Failure'], [success_count, num_episodes - success_count], 
        color=['#06A77D', '#E63946'])
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
ax1.text(0, success_count + 0.5, f'{success_count}/{num_episodes}\n({success_count/num_episodes*100:.1f}%)', 
         ha='center', fontsize=12, fontweight='bold')

# Steps per episode
ax2.bar(range(1, num_episodes + 1), all_steps, color='#2E86AB', alpha=0.7)
ax2.axhline(y=np.mean(all_steps), color='#E63946', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(all_steps):.1f}')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Steps', fontsize=12)
ax2.set_title('Steps per Episode', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Final distances
final_distances_mm = [all_distances[i][-1] * 1000 for i in range(num_episodes)]
colors_bar = ['#06A77D' if s else '#E63946' for s in all_success]
ax3.bar(range(1, num_episodes + 1), final_distances_mm, color=colors_bar, alpha=0.7)
ax3.axhline(y=5.0, color='black', linestyle='--', linewidth=2, label='Threshold (5mm)')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Final Distance (mm)', fontsize=12)
ax3.set_title('Final Distance to Goal', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Reward per episode
ax4.bar(range(1, num_episodes + 1), all_rewards, color='#F77F00', alpha=0.7)
ax4.axhline(y=np.mean(all_rewards), color='#E63946', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(all_rewards):.2f}')
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Total Reward', fontsize=12)
ax4.set_title('Reward per Episode', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved summary statistics plot")

# Calculate and save detailed statistics
stats = {
    'model': model_filename,
    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_episodes': num_episodes,
    'target_threshold_mm': env.target_threshold * 1000,
    'max_steps': env.max_steps,
    'results': {
        'success_rate': f"{success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)",
        'success_count': success_count,
        'total_episodes': num_episodes,
        'success_percentage': round(success_count/num_episodes*100, 2),
        'average_steps': round(float(np.mean(all_steps)), 2),
        'std_steps': round(float(np.std(all_steps)), 2),
        'min_steps': int(np.min(all_steps)),
        'max_steps': int(np.max(all_steps)),
        'average_reward': round(float(np.mean(all_rewards)), 2),
        'std_reward': round(float(np.std(all_rewards)), 2),
        'average_final_distance_mm': round(float(np.mean(final_distances_mm)), 3),
        'std_final_distance_mm': round(float(np.std(final_distances_mm)), 3),
        'min_final_distance_mm': round(float(np.min(final_distances_mm)), 3),
        'max_final_distance_mm': round(float(np.max(final_distances_mm)), 3),
    },
    'episode_details': [
        {
            'episode': i + 1,
            'goal': all_goals[i],
            'steps': all_steps[i],
            'reward': round(all_rewards[i], 2),
            'final_distance_mm': round(final_distances_mm[i], 3),
            'success': all_success[i]
        }
        for i in range(num_episodes)
    ]
}

# Save JSON
with open(output_dir / 'test_results.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("✓ Saved test_results.json")

# Save text summary
summary_text = f"""
{'='*80}
MODEL TEST RESULTS
{'='*80}

Model: {model_filename}
Test Date: {stats['test_date']}
Episodes: {num_episodes}
Target Threshold: {env.target_threshold * 1000}mm
Max Steps: {env.max_steps}

{'='*80}
OVERALL PERFORMANCE
{'='*80}

Success Rate:          {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)
Average Steps:         {stats['results']['average_steps']:.2f} ± {stats['results']['std_steps']:.2f}
Step Range:            {stats['results']['min_steps']} - {stats['results']['max_steps']}
Average Reward:        {stats['results']['average_reward']:.2f} ± {stats['results']['std_reward']:.2f}
Avg Final Distance:    {stats['results']['average_final_distance_mm']:.3f}mm ± {stats['results']['std_final_distance_mm']:.3f}mm
Distance Range:        {stats['results']['min_final_distance_mm']:.3f}mm - {stats['results']['max_final_distance_mm']:.3f}mm

{'='*80}
EPISODE DETAILS
{'='*80}

"""

for ep_detail in stats['episode_details']:
    success_marker = '✓' if ep_detail['success'] else '✗'
    summary_text += f"Episode {ep_detail['episode']:2d} {success_marker}: "
    summary_text += f"Steps={ep_detail['steps']:3d} | "
    summary_text += f"Reward={ep_detail['reward']:7.2f} | "
    summary_text += f"Dist={ep_detail['final_distance_mm']:6.3f}mm\n"

summary_text += f"\n{'='*80}\n"

with open(output_dir / 'test_summary.txt', 'w') as f:
    f.write(summary_text)

print("✓ Saved test_summary.txt")

# Print summary to console
print("\n" + summary_text)

# Clean up
env.close()
os.chdir(original_dir)

print(f"\n{'='*80}")
print(f"All results saved to: {output_dir}")
print(f"{'='*80}")
print("\nFiles created:")
print(f"  - {num_episodes} trajectory plots (trajectory_episode_*.png)")
print(f"  - distance_convergence_all_episodes.png")
print(f"  - summary_statistics.png")
print(f"  - test_results.json")
print(f"  - test_summary.txt")
print(f"\n{'='*80}")
