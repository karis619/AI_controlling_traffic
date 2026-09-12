import pygame
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Ensure the current script directory is on sys.path so local imports work when running this file directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the necessary RL library and the custom environment wrapper
from stable_baselines3 import DQN
try:
    # Prefer local import when comparison.py and environment.py are in the same folder
    from environment import CustomTrafficEnv
except Exception:
    # Fallback to package-style import if the project is installed as a package
    from ai_agent.environment import CustomTrafficEnv


# --- Configuration ---
# You must change this path to point to your saved DQN agent .zip file
TRAINED_MODEL_PATH = 'ai_agent/dqn_models/trained_dqn_agent.zip' 

# --- Helper to load and run the agent safely ---
def load_and_predict(agent, state):
    """Loads the agent and returns a deterministic action (0-3)."""
    try:
        # Predict the action using the loaded DQN agent
        # state must be wrapped in a batch dimension for the model
        action, _ = agent.predict(state, deterministic=True)
        return action.item()
    except Exception as e:
        # Fallback to simple fixed-cycle if the model fails (e.g., if it wasn't loaded)
        # This fallback prevents the whole demo from crashing
        print(f"Prediction Error: {e}. Using fixed cycle.")
        return 0 # Default to action 0

def run_comparison():
    """Runs a side-by-side simulation comparing DQN vs. Fixed Timing."""
    
    # 1. Initialize Environments
    # NOTE: Set render_mode="none" for speed if Pygame is not fully integrated.
    env_ai = CustomTrafficEnv(render_mode="none")
    env_normal = CustomTrafficEnv(render_mode="none")
    
    # 2. Load Trained Agent
    dqn_agent = None
    try:
        # Load the DQN model from the .zip file
        dqn_agent = DQN.load(TRAINED_MODEL_PATH, env=env_ai)
        print("✓ Loaded trained DQN agent successfully.")
    except Exception as e:
        print(f"✗ Model not found at '{TRAINED_MODEL_PATH}'. {e}. Using fixed cycle mode for AI.")

    
    print("\n" + "="*50)
    print("STARTING AI VS FIXED CYCLE COMPARISON")
    print("="*50)
    
    # 3. Setup Metrics and State
    max_steps = 2000 # Increase steps for better statistical results
    step_count = 0
    
    ai_metrics = {'total_reward': 0, 'vehicles_passed': 0, 'total_waiting_time': 0, 'emergency_stops': 0, 'congestion_level': 0, 'steps_with_congestion': 0}
    normal_metrics = {'total_reward': 0, 'vehicles_passed': 0, 'total_waiting_time': 0, 'emergency_stops': 0, 'congestion_level': 0, 'steps_with_congestion': 0}
    
    comparison_data = []
    
    state_ai, _ = env_ai.reset()
    state_normal, _ = env_normal.reset()
    
    # Pre-wrap states to match the DQN input shape (batch=1)
    state_ai = np.array(state_ai, dtype=np.float32)
    state_normal = np.array(state_normal, dtype=np.float32)
    
    running = True
    start_time = time.time()
    
    # 4. Main Comparison Loop
    while running and step_count < max_steps:
        
        # --- AI Mode Decision (DQN Agent) ---
        if dqn_agent:
            # DQN requires state input to be a batch, hence [state_ai]
            action_ai, _ = dqn_agent.predict(state_ai, deterministic=True)
            action_ai = int(action_ai)
        else:
            # Fallback (Simple Busiest Lane Rule for comparison)
            action_ai = np.argmax(state_ai)
            
        # --- Normal Mode Decision (Fixed Cycle) ---
        action_normal = step_count % env_normal.action_space.n # Cycle 0, 1, 2, 3, 0, 1, 2, 3...
        
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
        comparison_data.append({'step': step_count, 'ai_reward': reward_ai, 'normal_reward': reward_normal, 'ai_action': action_ai, 'normal_action': action_normal, 'timestamp': datetime.now()})
        
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
    ai_avg_waiting = ai_metrics['total_waiting_time'] / max(ai_metrics['vehicles_passed'], 1)
    normal_avg_waiting = normal_metrics['total_waiting_time'] / max(normal_metrics['vehicles_passed'], 1)
    
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
        f.write(f"AI Average Waiting Time: {ai_metrics['total_waiting_time'] / max(ai_metrics['vehicles_passed'], 1):.2f} steps\n")
        f.write(f"Normal Average Waiting Time: {normal_metrics['total_waiting_time'] / max(normal_metrics['vehicles_passed'], 1):.2f} steps\n")
        f.write(f"Total Simulation Steps: {total_steps}\n")
    
    print(f"\nResults saved to: {filepath}")

if __name__ == "__main__":
    run_comparison()