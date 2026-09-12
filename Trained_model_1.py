import cv2
import sys
import numpy as np
import os
import serial
import time
from gtts import gTTS
import pygame
from ultralytics import YOLO
from stable_baselines3 import DQN
import torch



pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h
pygame.display.quit()

# --- YOLO MODEL ---
MODEL_PATH = "vision/models/yolov8n_openvino_model/"
try:
    model = YOLO(MODEL_PATH)
    print("YOLO Model Loaded")
except Exception as e:
    print(f"YOLO Load Error: {e}")
    sys.exit()

# --- DQN BRAIN 
TRAINED_MODEL_PATH = r"D:\Adaptive_traffic\AI_TRAFFIC_CONTROL_PROJECT\ai_agent\dqn_models\dqn_traffic_brain_ULTIMATE.zip"
dqn_agent = None
try:
    dqn_agent = DQN.load(TRAINED_MODEL_PATH)
    print("DQN Brain Loaded - 16-dim observation space")
    
    # Extract the Q-network for direct timing calculations
    dqn_q_net = dqn_agent.q_net
    print("DQN Q-Network available for timing calculations")
    
except Exception as e:
    print(f"DQN Load Failed → Using Fallback: {e}")

# --- AUDIO - DUAL LANGUAGE ---
VOICE_EN_FILE = "walk_safe_en.mp3"
VOICE_SW_FILE = "walk_safe_sw.mp3"


if not os.path.exists(VOICE_EN_FILE):
    gTTS("It is now safe to cross the road.", lang='en').save(VOICE_EN_FILE)
if not os.path.exists(VOICE_SW_FILE):
    gTTS("Sasa ni salama kuvuka barabara.", lang='sw').save(VOICE_SW_FILE)

def play_walk_announcement():
    """Play English then Swahili announcement for pedestrians"""
    if not pygame.mixer.music.get_busy():
        # Play English first
        pygame.mixer.music.load(VOICE_EN_FILE)
        pygame.mixer.music.play()
        
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        pygame.mixer.music.load(VOICE_SW_FILE)
        pygame.mixer.music.play()

# --- ARDUINO ---
try:
    arduino = serial.Serial("COM4", 9600, timeout=1)
    time.sleep(2)
    print("Arduino Connected")
except:
    arduino = None
    print("Arduino Not Found → Simulation Mode")

# --- TIMING ---
YELLOW_TIME = 2.0
ALL_RED_TIME = 1.0
MIN_GREEN = 5.0
MAX_GREEN = 30.0
PED_TIME = 10.0
PED_THRESHOLD = 1


GREEN_CMD = ['1', '2', '3', '4']  # Arm 1, 2, 3, 4
YELLOW_CMD = ['q', 'w', 'e', 'r']  # Yellow for Arm 1, 2, 3, 4

def send(cmd):
    if arduino:
        print(f"→ {cmd}")
        arduino.reset_input_buffer()
        arduino.write(cmd.encode())
        time.sleep(0.1)
    else:
        print(f"[SIM] {cmd}")
        time.sleep(0.05)

def wait_ack():
    if not arduino: return
    t0 = time.time()
    while time.time() - t0 < 2:
        if arduino.in_waiting:
            if arduino.readline().decode().strip() == "OK":
                return
    print("ACK Timeout")

def transition(old, new):
    print(f"\nTRANSITION: {old} → {new}")
    if old in GREEN_CMD:
        # Convert command to arm index (1,2,3,4 → 0,1,2,3 for list indexing)
        idx = GREEN_CMD.index(old)
        send(YELLOW_CMD[idx])
        time.sleep(YELLOW_TIME)
    send('A')
    time.sleep(ALL_RED_TIME)
    send(new)
    wait_ack()
    if new == 'P':
        print("PEDESTRIANS CROSSING")
        play_walk_announcement()
        time.sleep(PED_TIME)
        send('A')
        time.sleep(ALL_RED_TIME)
        return 'A'
    return new

def get_dqn_green_time(cars, peds, wait_times, current_phase, selected_arm):
    """Get green time duration directly from DQN model based on current state"""
    if dqn_agent is None:
        # Fallback: basic calculation based on vehicle count
        base_time = MIN_GREEN
        extra_time = min(cars[selected_arm] * 0.8, 20)
        return min(base_time + extra_time, MAX_GREEN)
    
    try:
        
        state = []
        
        # 1. Vehicle counts for all 4 arms (4 values)
        state.extend(cars)
        
        # 2. Pedestrian counts for all 4 arms (4 values)  
        state.extend(peds)
        
        # 3. Wait times for all 4 arms (4 values)
        state.extend(wait_times)
        
        # 4. Current phase one-hot encoding (4 values)
        phase_one_hot = [1 if i == current_phase else 0 for i in range(4)]
        state.extend(phase_one_hot)
        
        # Total: 4 + 4 + 4 + 4 = 16 dimensions
        
        # Convert to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn_q_net(state_tensor).numpy()[0]
        
        # Use Q-value for the selected action to determine duration
        selected_q_value = q_values[selected_arm] if selected_arm < len(q_values) else q_values[0]
        
        # Normalize Q-value to time range
        max_q = np.max(q_values)
        min_q = np.min(q_values)
        
        if max_q == min_q:
            normalized = 0.5
        else:
            normalized = (selected_q_value - min_q) / (max_q - min_q)
        
        # Map to green time range
        base_green_time = MIN_GREEN + (normalized * (MAX_GREEN - MIN_GREEN))
        
        # Adjust based on immediate traffic needs (vehicle count on selected arm)
        vehicle_factor = min(cars[selected_arm] / 15.0, 2.0)  # Cap influence
        final_green_time = base_green_time * (1.0 + vehicle_factor * 0.4)
        
        return min(final_green_time, MAX_GREEN)
        
    except Exception as e:
        print(f"DQN timing calculation error: {e}")
        # Emergency fallback
        return min(MIN_GREEN + (cars[selected_arm] * 1.0), MAX_GREEN)

# --- CAMERA ---
URLS = {
    1: "http://10.224.246.139:5000/video_feed",  # Arm 1
    2: "http://10.224.246.254:8000/video_feed", # Arm 2
    3: "http://10.224.246.8:8002/video_feed",   # Arm 3
    4: "http://10.224.246.99:8001/video_feed"      # Arm 4
}

W = SCREEN_WIDTH // 2
H = SCREEN_HEIGHT // 2
ROI = {
    1: [10, 10, W-10, H-10],                    # Top Left - Arm 1
    2: [W+10, 10, SCREEN_WIDTH-10, H-10],       # Top Right - Arm 2  
    3: [10, H+10, W-10, SCREEN_HEIGHT-10],      # Bottom Left - Arm 3
    4: [W+10, H+10, SCREEN_WIDTH-10, SCREEN_HEIGHT-10]  # Bottom Right - Arm 4
}

caps = {}
for i, url in URLS.items():
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        caps[i] = cap
    else:
        print(f"Cam {i} Offline")

cv2.namedWindow("KENYA AI TRAFFIC", cv2.WINDOW_FULLSCREEN)

# --- STATE ---
current_light = 'A'
last_switch = time.time()
last_frame_time = time.time()
wait_times = [0.0, 0.0, 0.0, 0.0]  # For arms 1,2,3,4
green_start_time = time.time()
green_duration = MIN_GREEN
ped_voice_played = False

send('A')
print("KENYA AI 4-ARM TRAFFIC CONTROLLER STARTED")

try:
    while True:
        t = time.time()
        dt = t - last_frame_time
        last_frame_time = t

        # Read frames
        frames = {}
        for i in [1, 2, 3, 4]:  # Arms 1, 2, 3, 4
            if i in caps:
                ret, f = caps[i].read()
                if ret:
                    f = cv2.resize(f, (W, H))
                else:
                    f = np.zeros((H, W, 3), np.uint8)
                    cv2.putText(f, "NO SIGNAL", (50, H//2), 5, 2, (0,0,255), 3)
            else:
                f = np.zeros((H, W, 3), np.uint8)
                cv2.putText(f, f"ARM {i} OFFLINE", (50, H//2), 5, 2, (255,255,255), 3)
            frames[i] = f
        
        # 2x2 Matrix layout
        top_row = np.hstack((frames[1], frames[2]))
        bottom_row = np.hstack((frames[3], frames[4]))
        combined = np.vstack((top_row, bottom_row))
        
        # YOLO
        results = model.predict(combined, conf=0.3, verbose=False)[0]
        cars = [0, 0, 0, 0]  # Index 0: Arm 1, Index 1: Arm 2, Index 2: Arm 3, Index 3: Arm 4
        peds = [0, 0, 0, 0]  # Index 0: Arm 1, Index 1: Arm 2, Index 2: Arm 3, Index 3: Arm 4
        
        if results.boxes is not None:
            for box in results.boxes:
                label = model.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                for arm in [1, 2, 3, 4]:  # Check arms 1, 2, 3, 4
                    rx1, ry1, rx2, ry2 = ROI[arm]
                    if rx1 < cx < rx2 and ry1 < cy < ry2:
                        # Convert arm number to list index (1→0, 2→1, 3→2, 4→3)
                        arm_idx = arm - 1
                        if label in ["car", "truck", "bus", "motorcycle"]:
                            cars[arm_idx] += 1
                            cv2.rectangle(combined, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(combined, "CAR", (x1,y1-10), 0, 0.6, (0,255,0), 2)
                        elif label == "person":
                            peds[arm_idx] += 1
                            cv2.rectangle(combined, (x1,y1), (x2,y2), (0,0,255), 2)
                            cv2.putText(combined, "PED", (x1,y1-10), 0, 0.6, (0,0,255), 2)
                        break
        
        total_peds = sum(peds)
        # Convert current light to arm index (0,1,2,3 for arms 1,2,3,4)
        current_phase = GREEN_CMD.index(current_light) if current_light in GREEN_CMD else 0
        
        # Update wait times
        for i in range(4):  # i=0: Arm 1, i=1: Arm 2, i=2: Arm 3, i=3: Arm 4
            if cars[i] > 0 and i != current_phase:
                wait_times[i] += dt
            else:
                wait_times[i] = 0

        # === PEDESTRIAN PRIORITY ===
        if total_peds >= PED_THRESHOLD and current_light != 'P':
            current_light = transition(current_light, 'P')
            wait_times = [0, 0, 0, 0]
            last_switch = time.time()
            ped_voice_played = False

        # === DQN DECISION ===
        elif time.time() - last_switch > MIN_GREEN:
            selected_arm = 0
            
            if dqn_agent is None:
                # Fallback: select arm with most vehicles + wait time
                scores = [cars[i] + 0.1 * wait_times[i] for i in range(4)]
                selected_arm = np.argmax(scores)
                print(f"[FALLBACK] Arm {selected_arm + 1}")
            else:
                # Use DQN to select the best arm - with 16-dim state
                phase_one_hot = [1 if i == current_phase else 0 for i in range(4)]
                state_16 = cars + peds + wait_times + phase_one_hot  # 4+4+4+4=16
                
                # CORRECT: Use proper shape (16,) instead of (1,16)
                obs = np.array(state_16, dtype=np.float32)
                
                action, _ = dqn_agent.predict(obs, deterministic=True)
                selected_arm = int(action)
                
                if selected_arm not in [0, 1, 2, 3]:
                    scores = [cars[i] + 0.1 * wait_times[i] for i in range(4)]
                    selected_arm = np.argmax(scores)
                    print(f"[DQN] Fixed → Arm {selected_arm + 1}")
                
                print(f"[DQN] Arm {selected_arm + 1} | Cars: {cars} | Wait: {wait_times}")

            # Get green duration DIRECTLY from DQN brain
            green_duration = get_dqn_green_time(cars, peds, wait_times, current_phase, selected_arm)
            
            print(f"DQN Brain Decision: Arm {selected_arm + 1} → {green_duration:.1f}s (Vehicles: {cars[selected_arm]})")

            new_light = GREEN_CMD[selected_arm]
            if new_light != current_light:
                current_light = transition(current_light, new_light)
                green_start_time = time.time()
            last_switch = time.time()

        # === PED VOICE CONTROL ===
        if current_light == 'P' and not ped_voice_played:
            play_walk_announcement()
            ped_voice_played = True

        # === DISPLAY - 2x2 MATRIX ===
        # Arm 1 - Top Left
        cv2.putText(combined, f"ARM 1: {cars[0]} cars", (20, 60),
                    5, 0.8, (0,255,0), 2)
        # Arm 2 - Top Right
        cv2.putText(combined, f"ARM 2: {cars[1]} cars", (W + 20, 60),
                    5, 0.8, (0,255,0), 2)
        # Arm 3 - Bottom Left
        cv2.putText(combined, f"ARM 3: {cars[2]} cars", (20, H + 60),
                    5, 0.8, (0,255,0), 2)
        # Arm 4 - Bottom Right
        cv2.putText(combined, f"ARM 4: {cars[3]} cars", (W + 20, H + 60),
                    5, 0.8, (0,255,0), 2)

        # Green arm + DQN timer
        if current_light in GREEN_CMD:
            idx = GREEN_CMD.index(current_light)
            arm_number = idx + 1  # Convert to actual arm number
            elapsed = time.time() - green_start_time
            remaining = max(0.0, green_duration - elapsed)
            txt = f"ARM {arm_number} GREEN  {remaining:.1f}s (DQN)"
            cv2.putText(combined, txt, (20, SCREEN_HEIGHT - 120),
                        5, 1.4, (0,255,0), 4)
            
            # Add DQN info
            cv2.putText(combined, f"DQN BRAIN ACTIVE", (20, SCREEN_HEIGHT - 80),
                        5, 0.8, (255,255,0), 2)
                        
        elif current_light == 'P':
            cv2.putText(combined, "PEDESTRIAN GO", (20, SCREEN_HEIGHT - 120),
                        5, 1.4, (0,0,255), 4)
        else:
            cv2.putText(combined, "ALL RED", (20, SCREEN_HEIGHT - 120),
                        5, 1.4, (255,0,0), 4)

        cv2.putText(combined, f"PEDS: {total_peds}", (20, 120), 5, 1.2, (0,0,255), 3)
        cv2.putText(combined, f"LIGHT: {current_light}", (20, 180), 5, 1.2, (255,255,0), 3)


        cv2.imshow("KENYA AI TRAFFIC", combined)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    send('A')
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()