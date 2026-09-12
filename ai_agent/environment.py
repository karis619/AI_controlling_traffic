import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import time
import importlib.util

class CustomTrafficEnv(gym.Env):
    """
    Environment interface between Gym and your traffic simulation.
    Compatible with both training simulation and real physical prototype.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none"):
        super().__init__()

        # --- Locate simulation script (game_env.py) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        simulation_path = os.path.join(os.path.dirname(current_dir), "simulation")
        sim_file = os.path.join(simulation_path, "game_env.py")

        # --- Dynamically load simulation module ---
        if os.path.exists(sim_file):
            spec = importlib.util.spec_from_file_location("game_env", sim_file)
            self.simulation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.simulation_module)
            print("✅ Connected to game_env.py simulation")
        else:
            raise FileNotFoundError(f"❌ game_env.py not found at {sim_file}")

        # --- Enable RL mode (disables GUI) ---
        self.simulation_module.rl_mode = True
        print("🎮 RL headless mode enabled (for faster training)")

        # --- Define Observation & Action Spaces ---
        # 16 inputs = 4 lanes (vehicles, pedestrians, waiting time, green state)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)

        # 5 actions:
        # 0 = North-South Green
        # 1 = East-West Green
        # 2 = All Red (Transition)
        # 3 = Pedestrian Walk
        # 4 = Emergency / Special Mode
        self.action_space = spaces.Discrete(5)

        self.current_state = np.zeros(16, dtype=np.float32)
        self.step_counter = 0

        # Optional timing / simulation parameters
        self.max_steps = 2000

    def reset(self, seed=None, options=None):
        """Reset the simulation for a new training episode."""
        super().reset(seed=seed)
        try:
            # Reset simulation (custom method in game_env.py)
            self.current_state = np.array(self.simulation_module.reset_simulation(), dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Reset error: {e}, using zero state fallback")
            self.current_state = np.zeros(16, dtype=np.float32)

        self.step_counter = 0
        return self.current_state, {}

    def step(self, action):
        """
        Send action → simulation → receive next state, reward, done, info.
        """
        try:
            self.simulation_module.take_action(int(action))
            next_state = np.array(self.simulation_module.get_state(), dtype=np.float32)
            reward = float(self.simulation_module.calculate_reward())
            done = self.simulation_module.timeElapsed >= self.simulation_module.simTime

            info = {
                "vehicles_passed": self.simulation_module.get_vehicles_passed(),
                "waiting_time": self.simulation_module.get_total_waiting_time(),
                "congestion": self.simulation_module.get_congestion_level()
            }

            self.current_state = next_state
            self.step_counter += 1

            # Safety limit (prevent infinite training)
            if self.step_counter >= self.max_steps:
                done = True

            return next_state, reward, done, False, info

        except Exception as e:
            print(f"⚠️ Step error: {e}, using fallback safe step")
            return self.current_state, -1.0, True, False, {}

    def render(self):
        """Optional render method (unused in headless mode)."""
        if hasattr(self.simulation_module, "render") and not self.simulation_module.rl_mode:
            self.simulation_module.render()

    def close(self):
        """Ensure simulation cleans up properly."""
        if hasattr(self.simulation_module, "rl_mode"):
            self.simulation_module.rl_mode = False
        print("🧹 Simulation closed cleanly.")
