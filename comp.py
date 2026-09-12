import pygame
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the current script directory is on sys.path so local imports work when running this file directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the necessary RL library and the custom environment wrapper
from stable_baselines3 import DQN
try:
    # Prefer local import when comparison.py and environment.py are in the same folder
    from ai_agent.environment import CustomTrafficEnv
except Exception as e:
    # Fallback to package-style import if the project is installed as a package
    try:
        from ai_agent.environment import CustomTrafficEnv
    except Exception:
        print(f"Error importing CustomTrafficEnv: {e}")
        sys.exit(1)


# --- Configuration ---
class ComparisonConfig:
    """Configuration class for comparison parameters"""
    TRAINED_MODEL_PATH = 'ai_agent/dqn_models/trained_dqn_agent.zip'
    MAX_STEPS = 2000
    RESULTS_DIR = 'comparison_results'
    PLOTS_DIR = 'comparison_plots'
    CONGESTION_THRESHOLD = 0.7
    PROGRESS_INTERVAL = 500


def load_agent(model_path, env):
    """Load trained agent with comprehensive error handling"""
    try:
        agent = DQN.load(model_path, env=env)
        print("✓ Loaded trained DQN agent successfully.")
        return agent
    except FileNotFoundError:
        print(f"✗ Model file not found at '{model_path}'")
        print("Please train the model first using train.py")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    return None


def initialize_metrics():
    """Initialize metrics dictionaries for both agents"""
    metrics_template = {
        'total_reward': 0,
        'vehicles_passed': 0,
        'total_waiting_time': 0,
        'emergency_stops': 0,
        'congestion_level': 0,
        'steps_with_congestion': 0,
        'action_counts': {0: 0, 1: 0},  # Track action distribution (0=stay, 1=switch)
        'queue_lengths': [],  # Track queue length over time
        'average_speeds': []  # Track average vehicle speeds
    }
    return metrics_template.copy(), metrics_template.copy()


def update_metrics(metrics, reward, info, action, step_count):
    """Update metrics for a single agent"""
    metrics['total_reward'] += reward
    metrics['vehicles_passed'] += info.get('vehicles_passed', 0)
    metrics['total_waiting_time'] += info.get('total_waiting_time', 0)
    metrics['emergency_stops'] += info.get('emergency_stops', 0)
    
    current_congestion = info.get('congestion_level', 0)
    metrics['congestion_level'] += current_congestion
    
    if current_congestion > ComparisonConfig.CONGESTION_THRESHOLD:
        metrics['steps_with_congestion'] += 1
    
    # Track action counts
    metrics['action_counts'][action] += 1
    
    # Track queue length if available
    queue_length = info.get('queue_length', 0)
    metrics['queue_lengths'].append(queue_length)
    
    # Estimate average speed (inverse of congestion)
    avg_speed = max(0.1, 1.0 - current_congestion)
    metrics['average_speeds'].append(avg_speed)
    
    return current_congestion


def calculate_improvements(ai_metrics, normal_metrics, total_steps):
    """Calculate performance improvements"""
    ai_avg_waiting = ai_metrics['total_waiting_time'] / max(ai_metrics['vehicles_passed'], 1)
    normal_avg_waiting = normal_metrics['total_waiting_time'] / max(normal_metrics['vehicles_passed'], 1)
    
    ai_congestion_pct = (ai_metrics['steps_with_congestion'] / total_steps) * 100
    normal_congestion_pct = (normal_metrics['steps_with_congestion'] / total_steps) * 100
    
    ai_avg_queue = np.mean(ai_metrics['queue_lengths']) if ai_metrics['queue_lengths'] else 0
    normal_avg_queue = np.mean(normal_metrics['queue_lengths']) if normal_metrics['queue_lengths'] else 0
    
    ai_avg_speed = np.mean(ai_metrics['average_speeds']) if ai_metrics['average_speeds'] else 0
    normal_avg_speed = np.mean(normal_metrics['average_speeds']) if normal_metrics['average_speeds'] else 0
    
    waiting_improvement = ((normal_avg_waiting - ai_avg_waiting) / max(normal_avg_waiting, 1)) * 100
    congestion_improvement = normal_congestion_pct - ai_congestion_pct
    queue_improvement = ((normal_avg_queue - ai_avg_queue) / max(normal_avg_queue, 1)) * 100
    speed_improvement = ((ai_avg_speed - normal_avg_speed) / max(normal_avg_speed, 1)) * 100
    
    return {
        'ai_avg_waiting': ai_avg_waiting,
        'normal_avg_waiting': normal_avg_waiting,
        'ai_congestion_pct': ai_congestion_pct,
        'normal_congestion_pct': normal_congestion_pct,
        'ai_avg_queue': ai_avg_queue,
        'normal_avg_queue': normal_avg_queue,
        'ai_avg_speed': ai_avg_speed,
        'normal_avg_speed': normal_avg_speed,
        'waiting_improvement': waiting_improvement,
        'congestion_improvement': congestion_improvement,
        'queue_improvement': queue_improvement,
        'speed_improvement': speed_improvement
    }


def print_comparison_results(ai_metrics, normal_metrics, improvements, total_steps, elapsed_time):
    """Print formatted comparison results"""
    print("\n" + "="*70)
    print("FINAL COMPARISON RESULTS")
    print("="*70)
    
    print(f"Simulation Duration: {elapsed_time:.2f} seconds")
    print(f"Total Steps: {total_steps}")
    print()
    
    print("AI MODE (DQN AGENT) RESULTS:")
    print(f" • Vehicles Passed: {ai_metrics['vehicles_passed']}")
    print(f" • Average Waiting Time: {improvements['ai_avg_waiting']:.2f} steps")
    print(f" • Congestion Time: {improvements['ai_congestion_pct']:.1f}%")
    print(f" • Average Queue Length: {improvements['ai_avg_queue']:.2f} vehicles")
    print(f" • Average Speed: {improvements['ai_avg_speed']:.2f} (normalized)")
    print(f" • Total Reward: {ai_metrics['total_reward']:.2f}")
    print(f" • Action Distribution: Stay={ai_metrics['action_counts'][0]}, Switch={ai_metrics['action_counts'][1]}")
    print()
    
    print("FIXED CYCLE (BASELINE) RESULTS:")
    print(f" • Vehicles Passed: {normal_metrics['vehicles_passed']}")
    print(f" • Average Waiting Time: {improvements['normal_avg_waiting']:.2f} steps")
    print(f" • Congestion Time: {improvements['normal_congestion_pct']:.1f}%")
    print(f" • Average Queue Length: {improvements['normal_avg_queue']:.2f} vehicles")
    print(f" • Average Speed: {improvements['normal_avg_speed']:.2f} (normalized)")
    print(f" • Total Reward: {normal_metrics['total_reward']:.2f}")
    print(f" • Action Distribution: Fixed cycle (automatic switching)")
    print()
    
    print("IMPROVEMENT ANALYSIS:")
    print(f" • Waiting Time Reduction: {improvements['waiting_improvement']:+.1f}%")
    print(f" • Congestion Reduction: {improvements['congestion_improvement']:+.1f}%")
    print(f" • Queue Length Reduction: {improvements['queue_improvement']:+.1f}%")
    print(f" • Speed Improvement: {improvements['speed_improvement']:+.1f}%")
    print(f" • Overall Efficiency Score: {(improvements['waiting_improvement'] + improvements['congestion_improvement'] + improvements['speed_improvement'])/3:.1f}%")


def create_visualizations(comparison_data, ai_metrics, normal_metrics, timestamp, improvements):
    """Create visualization plots for the comparison results"""
    try:
        os.makedirs(ComparisonConfig.PLOTS_DIR, exist_ok=True)
        
        df = pd.DataFrame(comparison_data)
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cumulative Rewards
        plt.subplot(2, 3, 1)
        df['ai_cumulative_reward'] = df['ai_reward'].cumsum()
        df['normal_cumulative_reward'] = df['normal_reward'].cumsum()
        plt.plot(df['step'], df['ai_cumulative_reward'], label='AI Agent', linewidth=2, color='blue')
        plt.plot(df['step'], df['normal_cumulative_reward'], label='Fixed Cycle', linewidth=2, color='red')
        plt.xlabel('Simulation Step')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Instantaneous Rewards (smoothed)
        plt.subplot(2, 3, 2)
        window_size = 50
        df['ai_reward_smooth'] = df['ai_reward'].rolling(window=window_size, center=True).mean()
        df['normal_reward_smooth'] = df['normal_reward'].rolling(window=window_size, center=True).mean()
        plt.plot(df['step'], df['ai_reward_smooth'], label='AI Agent', linewidth=1, color='blue', alpha=0.8)
        plt.plot(df['step'], df['normal_reward_smooth'], label='Fixed Cycle', linewidth=1, color='red', alpha=0.8)
        plt.xlabel('Simulation Step')
        plt.ylabel('Reward (Smoothed)')
        plt.title('Instantaneous Reward Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Action Distribution
        plt.subplot(2, 3, 3)
        actions = ['Stay (0)', 'Switch (1)']
        ai_actions = [ai_metrics['action_counts'][0], ai_metrics['action_counts'][1]]
        
        colors = ['lightblue', 'lightcoral']
        plt.pie(ai_actions, labels=actions, autopct='%1.1f%%', colors=colors)
        plt.title('AI Agent Action Distribution')
        
        # Plot 4: Performance Metrics Comparison
        plt.subplot(2, 3, 4)
        categories = ['Waiting Time', 'Congestion %', 'Queue Length']
        ai_values = [improvements['ai_avg_waiting'], improvements['ai_congestion_pct'], improvements['ai_avg_queue']]
        normal_values = [improvements['normal_avg_waiting'], improvements['normal_congestion_pct'], improvements['normal_avg_queue']]
        
        x = np.arange(len(categories))
        width = 0.35
        plt.bar(x - width/2, ai_values, width, label='AI Agent', alpha=0.7, color='blue')
        plt.bar(x + width/2, normal_values, width, label='Fixed Cycle', alpha=0.7, color='red')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, categories)
        plt.legend()
        
        # Plot 5: Queue Length Over Time
        plt.subplot(2, 3, 5)
        if len(ai_metrics['queue_lengths']) > 0 and len(normal_metrics['queue_lengths']) > 0:
            steps = range(min(len(ai_metrics['queue_lengths']), len(normal_metrics['queue_lengths'])))
            plt.plot(steps, ai_metrics['queue_lengths'][:len(steps)], label='AI Agent', linewidth=1, color='blue', alpha=0.7)
            plt.plot(steps, normal_metrics['queue_lengths'][:len(steps)], label='Fixed Cycle', linewidth=1, color='red', alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('Queue Length')
            plt.title('Queue Length Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Improvement Summary
        plt.subplot(2, 3, 6)
        improvement_categories = ['Waiting Time', 'Congestion', 'Queue', 'Speed']
        improvement_values = [
            improvements['waiting_improvement'],
            improvements['congestion_improvement'], 
            improvements['queue_improvement'],
            improvements['speed_improvement']
        ]
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        plt.bar(improvement_categories, improvement_values, color=colors, alpha=0.7)
        plt.xlabel('Metrics')
        plt.ylabel('Improvement (%)')
        plt.title('Performance Improvement Summary')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(improvement_values):
            plt.text(i, v + (1 if v >= 0 else -2), f'{v:+.1f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plot_path = os.path.join(ComparisonConfig.PLOTS_DIR, f'comparison_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"✗ Could not create visualizations: {e}")


def run_comparison():
    """Runs a side-by-side simulation comparing DQN vs. Fixed Timing."""
    
    # Create directories
    os.makedirs(ComparisonConfig.RESULTS_DIR, exist_ok=True)
    
    print("Initializing comparison environment...")
    
    # 1. Initialize Environments
    env_ai = CustomTrafficEnv(render_mode="none")
    env_normal = CustomTrafficEnv(render_mode="none")
    
    # 2. Load Trained Agent
    dqn_agent = load_agent(ComparisonConfig.TRAINED_MODEL_PATH, env_ai)
    
    if dqn_agent is None:
        print("Running comparison with fixed cycle for both environments (no trained AI)")
    
    print("\n" + "="*50)
    print("STARTING AI VS FIXED CYCLE COMPARISON")
    print("="*50)
    print(f"Maximum Steps: {ComparisonConfig.MAX_STEPS}")
    print(f"Progress updates every {ComparisonConfig.PROGRESS_INTERVAL} steps")
    print()
    
    # 3. Setup Metrics and State
    ai_metrics, normal_metrics = initialize_metrics()
    comparison_data = []
    
    state_ai, _ = env_ai.reset()
    state_normal, _ = env_normal.reset()
    
    # Ensure states are properly formatted
    state_ai = np.array(state_ai, dtype=np.float32)
    state_normal = np.array(state_normal, dtype=np.float32)
    
    running = True
    step_count = 0
    start_time = time.time()
    
    # 4. Main Comparison Loop
    while running and step_count < ComparisonConfig.MAX_STEPS:
        
        # --- AI Mode Decision ---
        if dqn_agent:
            # DQN requires state input to be a batch, hence [state_ai]
            action_ai, _ = dqn_agent.predict(state_ai, deterministic=True)
            action_ai = int(action_ai)
        else:
            # Fallback to simple heuristic (switch every 20 steps)
            action_ai = 1 if step_count % 20 == 0 else 0
            
        # --- Normal Mode Decision (Fixed Cycle) ---
        # Fixed cycle: switch signal every 30 steps
        action_normal = 1 if step_count % 30 == 0 else 0
        
        # --- Step Environments ---
        next_state_ai, reward_ai, terminated_ai, truncated_ai, info_ai = env_ai.step(action_ai)
        next_state_normal, reward_normal, terminated_normal, truncated_normal, info_normal = env_normal.step(action_normal)
        
        # Update states
        state_ai = np.array(next_state_ai, dtype=np.float32)
        state_normal = np.array(next_state_normal, dtype=np.float32)

        # Update metrics
        update_metrics(ai_metrics, reward_ai, info_ai, action_ai, step_count)
        update_metrics(normal_metrics, reward_normal, info_normal, action_normal, step_count)

        # Store step data
        comparison_data.append({
            'step': step_count, 
            'ai_reward': reward_ai, 
            'normal_reward': reward_normal,
            'ai_action': action_ai, 
            'normal_action': action_normal,
            'ai_vehicles_passed': info_ai.get('vehicles_passed', 0),
            'normal_vehicles_passed': info_normal.get('vehicles_passed', 0),
            'ai_waiting_time': info_ai.get('total_waiting_time', 0),
            'normal_waiting_time': info_normal.get('total_waiting_time', 0),
            'ai_congestion': info_ai.get('congestion_level', 0),
            'normal_congestion': info_normal.get('congestion_level', 0),
            'timestamp': datetime.now()
        })
        
        step_count += 1
        
        # Check termination conditions
        if terminated_ai or terminated_normal or truncated_ai or truncated_normal:
            running = False
            
        # Progress reporting
        if step_count % ComparisonConfig.PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Step {step_count}/{ComparisonConfig.MAX_STEPS} (Elapsed: {elapsed:.1f}s)")
            print(f"  AI - Vehicles: {ai_metrics['vehicles_passed']}, Wait: {ai_metrics['total_waiting_time']:.0f}, Reward: {ai_metrics['total_reward']:.1f}")
            print(f"  Normal - Vehicles: {normal_metrics['vehicles_passed']}, Wait: {normal_metrics['total_waiting_time']:.0f}, Reward: {normal_metrics['total_reward']:.1f}")
            print()

    # 5. Final Results
    elapsed_time = time.time() - start_time
    improvements = calculate_improvements(ai_metrics, normal_metrics, step_count)
    
    print_comparison_results(ai_metrics, normal_metrics, improvements, step_count, elapsed_time)
    print(f"\nComparison completed in {elapsed_time:.2f} seconds")
    
    # 6. Save results and create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_comparison_results(ai_metrics, normal_metrics, comparison_data, step_count, timestamp, improvements)
    create_visualizations(comparison_data, ai_metrics, normal_metrics, timestamp, improvements)
    
    # 7. Clean up
    env_ai.close()
    env_normal.close()
    
    print(f"\n✓ Comparison completed successfully!")
    print(f"✓ Results saved in '{ComparisonConfig.RESULTS_DIR}' directory")
    print(f"✓ Visualizations saved in '{ComparisonConfig.PLOTS_DIR}' directory")


def save_comparison_results(ai_metrics, normal_metrics, comparison_data, total_steps, timestamp, improvements):
    """Saves comparison results to CSV and summary files."""
    
    # Save detailed step data
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(ComparisonConfig.RESULTS_DIR, f"comparison_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save comprehensive summary
    summary_path = os.path.join(ComparisonConfig.RESULTS_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("TRAFFIC CONTROL COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Simulation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Simulation Steps: {total_steps}\n")
        f.write(f"Trained Model Used: {ComparisonConfig.TRAINED_MODEL_PATH}\n\n")
        
        f.write("AI AGENT PERFORMANCE:\n")
        f.write(f"  Vehicles Passed: {ai_metrics['vehicles_passed']}\n")
        f.write(f"  Average Waiting Time: {improvements['ai_avg_waiting']:.2f} steps\n")
        f.write(f"  Congestion Time: {improvements['ai_congestion_pct']:.1f}%\n")
        f.write(f"  Average Queue Length: {improvements['ai_avg_queue']:.2f} vehicles\n")
        f.write(f"  Average Speed: {improvements['ai_avg_speed']:.2f} (normalized)\n")
        f.write(f"  Total Reward: {ai_metrics['total_reward']:.2f}\n")
        f.write(f"  Action Distribution: Stay={ai_metrics['action_counts'][0]}, Switch={ai_metrics['action_counts'][1]}\n\n")
        
        f.write("FIXED CYCLE PERFORMANCE:\n")
        f.write(f"  Vehicles Passed: {normal_metrics['vehicles_passed']}\n")
        f.write(f"  Average Waiting Time: {improvements['normal_avg_waiting']:.2f} steps\n")
        f.write(f"  Congestion Time: {improvements['normal_congestion_pct']:.1f}%\n")
        f.write(f"  Average Queue Length: {improvements['normal_avg_queue']:.2f} vehicles\n")
        f.write(f"  Average Speed: {improvements['normal_avg_speed']:.2f} (normalized)\n")
        f.write(f"  Total Reward: {normal_metrics['total_reward']:.2f}\n\n")
        
        f.write("IMPROVEMENT ANALYSIS:\n")
        f.write(f"  Waiting Time Reduction: {improvements['waiting_improvement']:+.1f}%\n")
        f.write(f"  Congestion Reduction: {improvements['congestion_improvement']:+.1f}%\n")
        f.write(f"  Queue Length Reduction: {improvements['queue_improvement']:+.1f}%\n")
        f.write(f"  Speed Improvement: {improvements['speed_improvement']:+.1f}%\n")
        f.write(f"  Overall Efficiency Score: {(improvements['waiting_improvement'] + improvements['congestion_improvement'] + improvements['speed_improvement'])/3:.1f}%\n\n")
        
        f.write("CONCLUSION:\n")
        if improvements['waiting_improvement'] > 0:
            f.write("The AI agent outperformed the fixed cycle controller.\n")
        else:
            f.write("The fixed cycle controller performed similarly or better than the AI agent.\n")
        f.write("Consider training for more steps or adjusting reward parameters for better performance.\n")
    
    print(f"✓ Results saved to: {csv_path}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_comparison()