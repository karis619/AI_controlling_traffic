import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
# --- IMPORT THE CALLBACK ---
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your new environment class
from Trafficenv import TrafficEnv

# --- 1. Create and check the environment ---
env = TrafficEnv()

# --- 2. SET UP THE CHECKPOINT CALLBACK ---
# This will save a model every 10,000 steps in a folder named "dqn_checkpoints"
# It will be named like "traffic_model_10000_steps.zip", "traffic_model_20000_steps.zip", etc.
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path='./dqn_checkpoints/',
  name_prefix='traffic_model'
)

# --- 3. Create the DQN "Brain" (Agent) ---
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_starts=1000, 
    buffer_size=50000,    
    tensorboard_log="./dqn_traffic_tensorboard/"
)

# --- 4. Train the Agent ---
print("Starting model training (with periodic checkpoint saving)...")
# We pass the callback to the .learn() method
model.learn(
    total_timesteps=100000, 
    log_interval=4, 
    callback=checkpoint_callback  # <-- PASS THE CALLBACK HERE
)

# --- 5. Save the FINAL Trained Model ---
# This line will still run when the *entire* training finishes successfully
model_save_path = "dqn_traffic_brain_FINAL"
model.save(model_save_path)

print(f"Training complete. Final model saved to {model_save_path}.zip")

# --- 6. How to use it (in your prototype) ---
print("Loading a saved model (e.g., a checkpoint)...")
# You can load the FINAL model or any of the checkpoints
# loaded_model = DQN.load("dqn_checkpoints/traffic_model_30000_steps.zip") 
# ... or ...
loaded_model = DQN.load(model_save_path)


# ... (rest of your testing code) ...
obs, info = env.reset()
for _ in range(100):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = env.step(action)
    print(f"State: {obs} -> Action: {action} -> Reward: {reward}")

env.sim.pygame.quit()