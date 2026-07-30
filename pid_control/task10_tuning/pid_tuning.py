"""
PID Controller Tuning and Testing
This module provides systematic testing of PID parameters to find optimal values
for the robot control system.
"""

import pybullet as p
import numpy as np
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Setup paths
datalabs_path = Path(__file__).parent.parent
sys.path.insert(0, str(datalabs_path))
from simulation.cloned.sim_class import Simulation

# Change to simulation directory
original_dir = Path.cwd()
sim_dir = datalabs_path / "simulation" / "cloned"
os.chdir(sim_dir)


class PID:
    """PID Controller implementation"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0
    
    def compute(self, error, dt=1.0):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class PIDTuner:
    """PID parameter tuning system"""
    
    def __init__(self, render=False):
        self.workspace_bounds = {
            'x': (-0.187, 0.253),
            'y': (-0.171, 0.220),
            'z': (0.170, 0.289)
        }
        self.results = []
        self.render = render
        self.sim = None
        
    def test_parameters(self, kp, ki, kd, num_trials=5, max_steps=1500, 
                       tolerance=0.001, max_velocity=0.5, visualize=False):
        """
        Test a set of PID parameters across multiple trials
        
        Args:
            kp, ki, kd: PID parameters to test
            num_trials: Number of different targets to test
            max_steps: Maximum steps per trial
            tolerance: Target tolerance in meters
            max_velocity: Maximum velocity constraint
            visualize: Whether to plot results
            
        Returns:
            Dict with performance metrics
        """
        print(f"\n{'='*60}")
        print(f"Testing: Kp={kp}, Ki={ki}, Kd={kd}")
        print(f"{'='*60}")
        
        # Create simulation once if not already created
        if self.sim is None:
            self.sim = Simulation(num_agents=1, render=self.render)
        
        trial_results = []
        
        for trial in range(num_trials):
            # Generate random target
            target = np.array([
                np.random.uniform(*self.workspace_bounds['x']),
                np.random.uniform(*self.workspace_bounds['y']),
                np.random.uniform(*self.workspace_bounds['z'])
            ])
            
            # Reset PID controllers
            pid_x = PID(kp, ki, kd)
            pid_y = PID(kp, ki, kd)
            pid_z = PID(kp, ki, kd)
            
            # Reset robot to start position (0, 0, 0.15)
            self.sim.set_start_position(0, 0, 0.15)
            
            # Track trajectory
            positions = []
            errors = []
            velocities = []
            robot_id = self.sim.robotIds[0]
            
            reached_target = False
            settling_step = max_steps
            
            for step in range(max_steps):
                # Get current position
                states = self.sim.get_states()
                pos = states[f'robotId_{robot_id}']['pipette_position']
                current_pos = np.array([pos[0], pos[1], pos[2]])
                
                positions.append(current_pos.copy())
                
                # Calculate error
                error = target - current_pos
                error_magnitude = np.linalg.norm(error)
                errors.append(error_magnitude)
                
                # Check if reached target
                if error_magnitude < tolerance and not reached_target:
                    reached_target = True
                    settling_step = step
                
                # Compute PID control
                vel_x = pid_x.compute(error[0])
                vel_y = pid_y.compute(error[1])
                vel_z = pid_z.compute(error[2])
                
                # Clip velocities
                vel_x = np.clip(vel_x, -max_velocity, max_velocity)
                vel_y = np.clip(vel_y, -max_velocity, max_velocity)
                vel_z = np.clip(vel_z, -max_velocity, max_velocity)
                
                velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                velocities.append(velocity_magnitude)
                
                # Apply control
                action = [vel_x, vel_y, vel_z, 0]
                self.sim.run([action], num_steps=1)
            
            # Calculate metrics
            final_error = errors[-1]
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            
            # Calculate overshoot (if any axis went past target significantly)
            positions_array = np.array(positions)
            overshoot = 0
            for i, axis in enumerate(['x', 'y', 'z']):
                axis_positions = positions_array[:, i]
                if len(axis_positions) > 10:
                    # Check if we overshot the target
                    target_val = target[i]
                    start_val = positions_array[0, i]
                    direction = np.sign(target_val - start_val)
                    
                    if direction != 0:
                        max_deviation = np.max(axis_positions * direction) if direction > 0 else np.min(axis_positions * direction)
                        target_in_direction = target_val * direction
                        if max_deviation > target_in_direction:
                            overshoot = max(overshoot, abs(max_deviation - target_in_direction))
            
            # Stability check (variance in last 100 steps)
            if len(errors) > 100:
                stability = np.std(errors[-100:])
            else:
                stability = np.std(errors)
            
            trial_result = {
                'trial': trial + 1,
                'target': target.tolist(),
                'final_error': final_error,
                'avg_error': avg_error,
                'max_error': max_error,
                'settling_step': settling_step,
                'reached_target': reached_target,
                'overshoot': overshoot,
                'stability': stability,
                'positions': positions_array,
                'errors': errors,
                'velocities': velocities
            }
            
            trial_results.append(trial_result)
            
            print(f"  Trial {trial+1}/{num_trials}: "
                  f"Final Error={final_error:.6f}m, "
                  f"Steps={settling_step if reached_target else 'DNF'}, "
                  f"Reached={'Yes' if reached_target else 'No'}")
        
        # Aggregate results
        success_rate = sum(1 for r in trial_results if r['reached_target']) / num_trials
        avg_settling_time = np.mean([r['settling_step'] for r in trial_results if r['reached_target']]) if success_rate > 0 else max_steps
        avg_final_error = np.mean([r['final_error'] for r in trial_results])
        avg_overshoot = np.mean([r['overshoot'] for r in trial_results])
        avg_stability = np.mean([r['stability'] for r in trial_results])
        
        summary = {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'success_rate': success_rate,
            'avg_settling_time': avg_settling_time,
            'avg_final_error': avg_final_error,
            'avg_overshoot': avg_overshoot,
            'avg_stability': avg_stability,
            'trials': trial_results
        }
        
        print(f"\nSummary:")
        print(f"  Success Rate: {success_rate*100:.1f}%")
        print(f"  Avg Settling Time: {avg_settling_time:.1f} steps")
        print(f"  Avg Final Error: {avg_final_error:.6f}m")
        print(f"  Avg Overshoot: {avg_overshoot:.6f}m")
        print(f"  Avg Stability: {avg_stability:.6f}")
        
        if visualize:
            self._visualize_results(summary)
        
        self.results.append(summary)
        return summary
    
    def _visualize_results(self, summary):
        """Visualize the results of a parameter test"""
        trials = summary['trials']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Error over time for all trials
        ax1 = plt.subplot(2, 3, 1)
        for i, trial in enumerate(trials):
            ax1.plot(trial['errors'], alpha=0.6, label=f"Trial {i+1}")
        ax1.axhline(y=0.001, color='r', linestyle='--', label='Tolerance')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Error (m)')
        ax1.set_title(f'Error vs Time\nKp={summary["kp"]}, Ki={summary["ki"]}, Kd={summary["kd"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Velocity profiles
        ax2 = plt.subplot(2, 3, 2)
        for i, trial in enumerate(trials):
            ax2.plot(trial['velocities'], alpha=0.6, label=f"Trial {i+1}")
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Profiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 3D trajectory (first trial)
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        if trials:
            trial = trials[0]
            positions = trial['positions']
            target = trial['target']
            ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'b-', alpha=0.6, label='Trajectory')
            ax3.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                       c='g', s=100, marker='o', label='Start')
            ax3.scatter([target[0]], [target[1]], [target[2]], 
                       c='r', s=100, marker='*', label='Target')
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            ax3.set_title('3D Trajectory (Trial 1)')
            ax3.legend()
        
        # Plot 4: Final errors distribution
        ax4 = plt.subplot(2, 3, 4)
        final_errors = [trial['final_error'] for trial in trials]
        ax4.bar(range(1, len(final_errors) + 1), final_errors)
        ax4.axhline(y=0.001, color='r', linestyle='--', label='Tolerance')
        ax4.set_xlabel('Trial')
        ax4.set_ylabel('Final Error (m)')
        ax4.set_title('Final Errors by Trial')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Settling times
        ax5 = plt.subplot(2, 3, 5)
        settling_times = [trial['settling_step'] if trial['reached_target'] else 1500 
                         for trial in trials]
        colors = ['g' if trial['reached_target'] else 'r' for trial in trials]
        ax5.bar(range(1, len(settling_times) + 1), settling_times, color=colors)
        ax5.set_xlabel('Trial')
        ax5.set_ylabel('Settling Time (steps)')
        ax5.set_title('Settling Times\n(Green=Success, Red=Failed)')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary metrics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
        PID Parameters:
        Kp = {summary['kp']}
        Ki = {summary['ki']}
        Kd = {summary['kd']}
        
        Performance Metrics:
        Success Rate: {summary['success_rate']*100:.1f}%
        Avg Settling Time: {summary['avg_settling_time']:.1f} steps
        Avg Final Error: {summary['avg_final_error']:.6f} m
        Avg Overshoot: {summary['avg_overshoot']:.6f} m
        Avg Stability: {summary['avg_stability']:.6f}
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    def cleanup(self):
        """Close the simulation properly"""
        if self.sim is not None:
            self.sim.close()
            self.sim = None
    
    def grid_search(self, kp_range, ki_range, kd_range, num_trials=3):
        """
        Perform grid search over parameter ranges
        
        Args:
            kp_range: List of Kp values to test
            ki_range: List of Ki values to test
            kd_range: List of Kd values to test
            num_trials: Number of trials per parameter set
        """
        print(f"\n{'='*60}")
        print("Starting Grid Search")
        print(f"Kp range: {kp_range}")
        print(f"Ki range: {ki_range}")
        print(f"Kd range: {kd_range}")
        print(f"Total combinations: {len(kp_range) * len(ki_range) * len(kd_range)}")
        print(f"{'='*60}\n")
        
        for kp in kp_range:
            for ki in ki_range:
                for kd in kd_range:
                    self.test_parameters(kp, ki, kd, num_trials=num_trials)
        
        # Find best parameters
        best_result = max(self.results, 
                         key=lambda x: (x['success_rate'], 
                                       -x['avg_settling_time'], 
                                       -x['avg_final_error']))
        
        print(f"\n{'='*60}")
        print("BEST PARAMETERS FOUND:")
        print(f"{'='*60}")
        print(f"Kp = {best_result['kp']}")
        print(f"Ki = {best_result['ki']}")
        print(f"Kd = {best_result['kd']}")
        print(f"Success Rate: {best_result['success_rate']*100:.1f}%")
        print(f"Avg Settling Time: {best_result['avg_settling_time']:.1f} steps")
        print(f"Avg Final Error: {best_result['avg_final_error']:.6f}m")
        print(f"{'='*60}\n")
        
        return best_result
    
    def save_results(self, filename='pid_tuning_results.json'):
        """Save tuning results to JSON file"""
        output_path = Path(__file__).parent / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            for trial in serializable_result['trials']:
                trial['positions'] = trial['positions'].tolist()
                trial['errors'] = [float(e) for e in trial['errors']]
                trial['velocities'] = [float(v) for v in trial['velocities']]
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def compare_results(self):
        """Compare all tested parameter sets"""
        if not self.results:
            print("No results to compare yet!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        params = [f"Kp={r['kp']}\nKi={r['ki']}\nKd={r['kd']}" for r in self.results]
        success_rates = [r['success_rate'] * 100 for r in self.results]
        settling_times = [r['avg_settling_time'] for r in self.results]
        final_errors = [r['avg_final_error'] for r in self.results]
        overshoots = [r['avg_overshoot'] for r in self.results]
        
        # Plot 1: Success rates
        axes[0, 0].bar(range(len(params)), success_rates)
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_xticks(range(len(params)))
        axes[0, 0].set_xticklabels(params, rotation=45, ha='right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Settling times
        axes[0, 1].bar(range(len(params)), settling_times)
        axes[0, 1].set_ylabel('Avg Settling Time (steps)')
        axes[0, 1].set_title('Settling Time Comparison')
        axes[0, 1].set_xticks(range(len(params)))
        axes[0, 1].set_xticklabels(params, rotation=45, ha='right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Final errors
        axes[1, 0].bar(range(len(params)), final_errors)
        axes[1, 0].set_ylabel('Avg Final Error (m)')
        axes[1, 0].set_title('Final Error Comparison')
        axes[1, 0].set_xticks(range(len(params)))
        axes[1, 0].set_xticklabels(params, rotation=45, ha='right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Overshoots
        axes[1, 1].bar(range(len(params)), overshoots)
        axes[1, 1].set_ylabel('Avg Overshoot (m)')
        axes[1, 1].set_title('Overshoot Comparison')
        axes[1, 1].set_xticks(range(len(params)))
        axes[1, 1].set_xticklabels(params, rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main tuning script"""
    tuner = PIDTuner(render=False)  # Set render=True to visualize, False for faster testing
    
    try:
        # Option 1: Test specific parameters with detailed visualization
        print("\n=== Testing Current Parameters ===")
        tuner.test_parameters(kp=15, ki=0, kd=3, num_trials=5, visualize=True)
        
        # Option 2: Quick grid search around current values
        print("\n=== Grid Search ===")
        tuner.grid_search(
            kp_range=[10, 15, 20],
            ki_range=[0, 0.1, 0.5],
            kd_range=[1, 3, 5],
            num_trials=3
        )
        
        # Compare all results
        tuner.compare_results()
        
        # Save results
        tuner.save_results()
    
    finally:
        # Always cleanup
        tuner.cleanup()


if __name__ == "__main__":
    main()
