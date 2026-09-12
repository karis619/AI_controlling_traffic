import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# --- IMPORTANT ---
# Import your *existing* simulation class
from game_env import Simulation, Config

# Define how many SIMULATION TICKS one 'step' lasts.
# If your FPS is 60, 5 seconds = 300 ticks.
SIM_STEP_DURATION_TICKS = 5 * Config.FPS # 5 seconds * 60 FPS = 300 ticks

class TrafficEnv(gym.Env):
    """
    A custom Gymnasium environment for the traffic simulation.
    This class *wraps* your existing Simulation class.
    
    *** THIS VERSION IS MODIFIED FOR HIGH-SPEED (VIRTUAL TIME) TRAINING ***
    """
    
    # Add render_mode for Gymnasium compatibility
    metadata = {"render_modes": ["human"], "render_fps": Config.FPS}
    
    def __init__(self, render_mode=None):
        super(TrafficEnv, self).__init__()
        
        self.sim = Simulation()
        
        # --- 1. Define Action Space ---
        # (Switch to arm 0, 1, 2, or 3)
        self.action_space = spaces.Discrete(4)
        
        # --- 2. Define Observation Space (State) ---
        # 4 car counts, 4 ped counts, 4 for current green, 1 for elapsed time
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # --- 3. Handle Rendering ---
        self.render_mode = render_mode
        
        # --- 4. Virtual Clock ---
        self.time_since_last_switch = 0.0

    def _get_obs(self):
        """Helper function to get the 13-vector state from the simulation."""
        
        # 1. Get car/ped counts (8 values)
        car_counts = []
        ped_counts = []
        for i in range(4):
            signal = self.sim.traffic_manager.get_signal(i)
            car_counts.append(signal.cars)
            ped_counts.append(signal.peds)
            
        # 2. Get current green light (4 values)
        current_green_arm, _, _ = self.sim.traffic_manager.get_state()
        green_one_hot = np.zeros(4)
        green_one_hot[current_green_arm] = 1
        
        # 3. Get elapsed time (1 value)
        # Use our virtual clock, not the real-time clock
        elapsed_time = self.time_since_last_switch

        # Concatenate all parts into one flat array
        obs = np.concatenate(
            (car_counts, ped_counts, green_one_hot, [elapsed_time])
        ).astype(np.float32)
        
        return obs

    def _get_reward(self):
        """Helper function to calculate the reward."""
        
        # Reward is the *negative* total waiting cars and pedestrians
        # This punishes the agent for long queues
        total_wait = 0
        for i in range(4):
            signal = self.sim.traffic_manager.get_signal(i)
            total_wait += signal.cars
            total_wait += signal.peds
            
        return -total_wait

    def reset(self, seed=None, options=None):
        """Called at the start of each new episode."""
        super().reset(seed=seed)
        
        # Reset your simulation logic
        self.sim = Simulation()
        
        # Manually run a single frame to initialize
        self.sim._handle_input()
        self.sim._update_state() # This will get the initial counts
        
        # Reset virtual clock
        self.time_since_last_switch = 0.0
        
        if self.render_mode == "human":
            self.sim._render()
        
        observation = self._get_obs()
        info = {} # We don't need to return extra info
        
        return observation, info

    def step(self, action):
        """
        The main loop. The agent picks an 'action' (0-3).
        We "FAST-FORWARD" the simulation for SIM_STEP_DURATION_TICKS.
        """
        
        # --- 1. Apply the Action ---
        current_green_arm, _, _ = self.sim.traffic_manager.get_state()
        
        if action != current_green_arm:
            # The agent wants to switch lights.
            # (Using your placeholder logic for now)
            tm = self.sim.traffic_manager
            tm.lock.acquire()
            tm.currentGreen = action
            tm.nextGreen = (action + 1) % tm.num_signals
            tm.lock.release()
            
            # Reset the timer for this new phase
            self.time_since_last_switch = 0.0
            
            # --- TODO: Implement Yellow Light Logic ---
            # This is where you would run a *fast-forward* yellow light
            # cycle (e.g., run self.sim._update_state() 120 times for 2 sec)
            # For now, we are just switching instantly.


        # --- 2. Run Simulation for a Fixed Time (THE FAST WAY) ---
        # We replace the 5-second "while" loop with an instant "for" loop.
        # This "fast-forwards" the simulation.
        
        for _ in range(SIM_STEP_DURATION_TICKS):
            # NO RENDERING. NO CLOCK.TICK(). Just the logic.
            self.sim._handle_input()
            self.sim._update_state() # This updates car/ped counts
            
            # Update virtual clock (1 tick = 1 / FPS virtual seconds)
            self.time_since_last_switch += (1.0 / Config.FPS)

        # --- 3. (Optional) Render one frame if in "human" mode ---
        if self.render_mode == "human":
            self.sim._render()
            self.sim.clock.tick(Config.FPS) # Slow it down *only* if watching

        # --- 4. Get results ---
        observation = self._get_obs()
        reward = self._get_reward()
        
        terminated = False # Game doesn't end
        truncated = False  # No time limit
        info = {}

        return observation, reward, terminated, truncated, info