import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import CustomTrafficEnv

def train_model():
    print("🚦 Initializing DQN training for traffic AI")

    # --- Directories ---
    MODELS_DIR = "ai_agent/dqn_models"
    LOG_DIR = "logs/dqn_training"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Environment ---
    env = CustomTrafficEnv(render_mode="none")

    # --- Model Setup ---
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        tensorboard_log=LOG_DIR
    )

    # --- Callback for periodic saves ---
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=MODELS_DIR,
        name_prefix="dqn_traffic"
    )

    # --- Training ---
    TIMESTEPS = 50000
    print(f"🎯 Training for {TIMESTEPS} timesteps...")
    start_time = time.time()

    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    elapsed = time.time() - start_time
    print(f"✅ Training finished in {elapsed:.1f}s")

    # --- Save final model ---
    model_path = os.path.join(MODELS_DIR, "trained_dqn_agent.zip")
    model.save(model_path)
    print(f"🎉 Model saved at {model_path}")

    env.close()

if __name__ == "__main__":
    train_model()
