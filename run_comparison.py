# This file is: /run_comparison.py
#
# This script loads your custom PyTorch .pth file and
# compares it against a fixed-cycle light.

import pygame
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Ensure the 'ai_agent' folder is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.join(current_dir, 'ai_agent')
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

try:
    # Import the missing dependencies
    from ai_agent.environment import CustomTrafficEnv
    from ai_agent.train import DQNAgent
except ImportError as e:
    print(f"Error: Could not import dependencies. {e}")
    print("Make sure 'ai_agent/train.py' and 'ai_agent/environment.py' exist.")
    sys.exit()


# --- Configuration ---
# This path points to the .pth file you provided
TRAINED_MODEL_PATH = 'ai_agent/ai_agent/models/simple_traffic_dqn_final.pth' 

def run_comparison():
    """Runs a side-by-side simulation comparing DQN vs. Fixed Timing."""
    
    # 1. Initialize Environments
    env_ai = CustomTrafficEnv(render_mode="none")
    env_normal = CustomTrafficEnv(render_mode="none")
    
    # 2. Load Trained Agent
    state_size = env_ai.observation_space.shape[0]
    action_size = env_ai.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    try:
        # --- THIS IS WHERE THE .pth FILE IS USED ---
        agent.load(TRAINED_MODEL_PATH)
        print(f"✓ Loaded trained model successfully from {TRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"✗ No trained model found: {e}. Using random agent for demo.")
    
    # Set epsilon to 0 for deterministic (non-random) behavior
    agent.epsilon = 0.0
    
    print("\n" + "="*50)
    print("STARTING AI VS FIXED CYCLE COMPARISON")
    print("="*50)
    
    # 3. Setup Metrics and State
    max_steps = 2000
    step_count = 0
    
    ai_metrics = {'total_reward': 0, 'vehicles_passed': 0, 'total_waiting_time': 0, 'emergency_stops': 0, 'congestion_level': 0, 'steps_with_congestion': 0}
    normal_metrics = {'total_reward': 0, 'vehicles_passed': 0, 'total_waiting_time': 0, 'emergency_stops': 0, 'congestion_level': 0, 'steps_with_congestion': 0}
    
    comparison_data = []
    
    state_ai, _ = env_ai.reset()
    state_normal, _ = env_normal.reset()
    
    state_ai = np.array(state_ai, dtype=np.float32)
    state_normal = np.array(state_normal, dtype=np.float32)
    
    running = True
    start_time = time.time()
    
    # 4. Main Comparison Loop
    while running and step_count < max_steps:
        
        # --- AI Mode Decision (DQN Agent) ---
        action_ai = agent.act(state_ai, eps=0.0) # Use eps=0.0 for pure exploitation
            
        # --- Normal Mode Decision (Fixed Cycle) ---
        action_normal = step_count % env_normal.action_space.n # Cycle 0, 1, 2, 3...
        
        # --- Step Environments ---
        next_state_ai, reward_ai, terminated_ai, truncated_ai, info_ai = env_ai.step(action_ai)
        next_state_normal, reward_normal, terminated_normal, truncated_normal, info_normal = env_normal.step(action_normal)
        
        # Update states
        state_ai = np.array(next_state_ai, dtype=np.float32)
        state_normal = np.array(next_state_normal, dtype=np.float32)

        # Update AI Metrics
        ai_metrics['total_reward'] += reward_ai
        ai_metrics['vehicles_passed'] += info_ai.get('vehicles_passed', 0)
        ai_metrics['total_waiting_time'] += info_ai.get('total_waiting_time', 0)
        current_congestion_ai = info_ai.get('congestion_level', 0)
        ai_metrics['congestion_level'] += current_congestion_ai
        if current_congestion_ai > 0.7: ai_metrics['steps_with_congestion'] += 1

        # Update Normal Metrics
        normal_metrics['total_reward'] += reward_normal
        normal_metrics['vehicles_passed'] += info_normal.get('vehicles_passed', 0)
        normal_metrics['total_waiting_time'] += info_normal.get('total_waiting_time', 0)
        current_congestion_normal = info_normal.get('congestion_level', 0)
        normal_metrics['congestion_level'] += current_congestion_normal
        if current_congestion_normal > 0.7: normal_metrics['steps_with_congestion'] += 1

        # Store step data
        comparison_data.append({'step': step_count, 'ai_reward': reward_ai, 'normal_reward': reward_normal, 'ai_action': action_ai, 'normal_action': action_normal})
        
        step_count += 1
        
        if terminated_ai or terminated_normal:
            running = False
            
        # 5. Print Progress
        if step_count % 500 == 0:
            print(f"Step {step_count}/{max_steps}")
            print(f"AI - Total Wait: {ai_metrics['total_waiting_time']:.0f}, Vehicles: {ai_metrics['vehicles_passed']}")
            print(f"Normal - Total Wait: {normal_metrics['total_waiting_time']:.0f}, Vehicles: {normal_metrics['vehicles_passed']}")

    # 6. Final Results Analysis
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    # Calculate averages
    ai_avg_waiting = ai_metrics['total_waiting_time'] / max(step_count, 1)
    normal_avg_waiting = normal_metrics['total_waiting_time'] / max(step_count, 1)
    
    ai_congestion_percentage = (ai_metrics['steps_with_congestion'] / step_count) * 100
    normal_congestion_percentage = (normal_metrics['steps_with_congestion'] / step_count) * 100
    
    print("AI MODE (DQN AGENT) RESULTS:")
    print(f" • Vehicles Passed: {ai_metrics['vehicles_passed']}")
    print(f" • Average Waiting Time: {ai_avg_waiting:.2f} steps (Lower is better)")
    print(f" • Congestion Time: {ai_congestion_percentage:.1f}%")
    print()
    
    print("FIXED MODE (BASELINE) RESULTS:")
    print(f" • Vehicles Passed: {normal_metrics['vehicles_passed']}")
    print(f" • Average Waiting Time: {normal_avg_waiting:.2f} steps")
    print(f" • Congestion Time: {normal_congestion_percentage:.1f}%")
    print()
    
    # Improvement calculations
    waiting_improvement = ((normal_avg_waiting - ai_avg_waiting) / max(normal_avg_waiting, 1)) * 100
    
    print("IMPROVEMENT ANALYSIS:")
    print(f" • Waiting Time Reduction: {waiting_improvement:+.1f}%")
    print(f" • Congestion Reduction: {normal_congestion_percentage - ai_congestion_percentage:+.1f}%")
    
    # 7. Save results
    save_comparison_results(ai_metrics, normal_metrics, comparison_data, step_count)
    
    env_ai.close()
    env_normal.close()

def save_comparison_results(ai_metrics, normal_metrics, comparison_data, total_steps):
    """Saves comparison results to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results_{timestamp}.csv"
    
    os.makedirs('comparison_results', exist_ok=True)
    filepath = os.path.join('comparison_results', filename)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(filepath, index=False)
    
    summary_filepath = os.path.join('comparison_results', f"summary_{timestamp}.txt")
    with open(summary_filepath, 'w') as f:
        f.write("TRAFFIC CONTROL COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"AI Average Waiting Time: {ai_metrics['total_waiting_time'] / max(total_steps, 1):.2f} steps\n")
        f.write(f"Normal Average Waiting Time: {normal_metrics['total_waiting_time'] / max(total_steps, 1):.2f} steps\n")
        f.write(f"Total Simulation Steps: {total_steps}\n")
    
    print(f"\nResults saved to: {filepath}")

if __name__ == "__main__":
    run_comparison()