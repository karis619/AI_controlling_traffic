import os
import time
import numpy as np
from stable_baselines3 import DQN
from ai_agent.environment import CustomTrafficEnv
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback

# --- 1. Setup (Creates necessary folders) ---
MODELS_DIR = "ai_agent/dqn_models"
LOG_DIR = "logs/dqn_training"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 2. Create Environment ---
env = CustomTrafficEnv(render_mode="none")
print("✅ Headless training environment created.")

# --- 3. Setup Automatic Model Checkpoints ---
# Saves model every 25,000 timesteps to prevent progress loss
checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=MODELS_DIR,
    name_prefix="dqn_traffic_checkpoint"
)

# --- 4. Initialize the DQN Agent ---
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.0003,         # Slightly lower → more stable
    buffer_size=100000,           # Replay buffer
    learning_starts=1000,         # Start training after collecting 1k samples
    batch_size=64,                # Larger batch for smoother learning
    gamma=0.99,                   # Discount factor (future reward importance)
    exploration_fraction=0.2,     # More exploration at the start
    exploration_final_eps=0.02,   # Slightly more random actions during training
    train_freq=4,
    gradient_steps=2,
    target_update_interval=1000   # Update target network less often for stability
)

print("🧠 DQN Model initialized. Starting training...")

# --- 5. Train Model ---
TIMESTEPS_TO_TRAIN = 200000  # Train longer for better performance
model.learn(
    total_timesteps=TIMESTEPS_TO_TRAIN,
    log_interval=10,
    callback=checkpoint_callback
)

# --- 6. Save Final Model ---
model_filename = "my_traffic_agent_final.zip"
model_path = os.path.join(MODELS_DIR, model_filename)
model.save(model_path)

print("\n✅ --- Training Complete ---")
print(f"Model saved to: {model_path}")

# --- 7. Clean up ---
env.close()
print("🧹 Environment closed successfully.")
