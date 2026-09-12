import cv2
import sys
import numpy as np
import os
from ultralytics import YOLO
import pygame 
import serial
import time
from gtts import gTTS

# --- 1. INITIALIZE ENVIRONMENT & AUDIO ---
pygame.init()
pygame.mixer.init()
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h
print(f"Screen detected: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
pygame.display.quit() 

# --- 2. YOLO "EYES" CONFIGURATION ---
MODEL_PATH = "vision/models/yolov8n_openvino_model/"
LABELS_TO_COUNT = {"car", "bus", "truck", "motorcycle", "bicycle", "person"}

print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Successfully loaded OpenVINO model.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- 3. VOICE & ARDUINO SETUP ---
VOICE_FILE_WALK = "voice_walk.mp3"
if not os.path.exists(VOICE_FILE_WALK):
    tts = gTTS(text="It is now safe to cross the road.", lang='en')
    tts.save(VOICE_FILE_WALK)

def play_voice_command(file_path):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

try:
    ARDUINO_PORT = "COM4" 
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=9600, timeout=0.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)
except Exception as e:
    print(f"--- WARNING: COULD NOT CONNECT TO ARDUINO ---")
    arduino = None

def wait_for_arduino_ack(arduino_serial, timeout=3):
    if arduino_serial is None:
        return True
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if arduino_serial.in_waiting > 0:
            line = arduino_serial.readline().decode('utf-8').strip()
            if line == "OK":
                return True
    return False

# --- 4. CAMERA STREAM CONFIGURATION ---
URLS = { 1: "http://192.168.2.103:8001/video_feed", 2: "http://192.168.2.102:8000/video_feed", 3: "http://192.168.2.100:8002/video_feed" }
TILE_WIDTH = SCREEN_WIDTH // 2
TILE_HEIGHT = SCREEN_HEIGHT // 2
black_frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
cv2.putText(black_frame, "OFFLINE", (TILE_WIDTH // 2 - 70, TILE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# --- 5. CONNECT TO CAMERAS ---
caps = {}
for (cam_index, url) in URLS.items():
    print(f"Attempting to connect to camera {cam_index} at {url}...")
    if str(url).startswith("local_cam_"):
        try: device_id = int(url.split("_")[-1]); caps[cam_index] = cv2.VideoCapture(device_id)
        except ValueError: continue
    else: caps[cam_index] = cv2.VideoCapture(url)

    if not caps[cam_index].isOpened():
        print(f"!!! Camera {cam_index} is OFFLINE. Check firewall/IP. !!!")
        if cam_index in caps: del caps[cam_index] 
        
print(f"\nSuccessfully opened {len(caps)} camera stream(s).")

# --- 6. ROI & LOGIC SETUP ---
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
PEDESTRIAN_LABEL = "person"
PEDESTRIAN_THRESHOLD = 3
current_green_light = 0 
last_action_sent = '0'

CAR_ROIS = { 0: [10, 10, TILE_WIDTH - 10, TILE_HEIGHT - 10], 1: [TILE_WIDTH + 10, 10, (TILE_WIDTH * 2) - 10, TILE_HEIGHT - 10], 2: [10, TILE_HEIGHT + 10, TILE_WIDTH - 10, (TILE_HEIGHT * 2) - 10], 3: [TILE_WIDTH + 10, TILE_HEIGHT + 10, (TILE_WIDTH * 2) - 10, (TILE_HEIGHT * 2) - 10] }
PEDESTRIAN_ROIS = CAR_ROIS

# --- 7. MAIN LOGIC AND LOOP ---
WINDOW_NAME = "AI Traffic Control - Fullscreen Demo"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def get_counts_and_process_frame(frame):
    # This function is run inside the loop to avoid clutter
    results = model.predict(frame, verbose=False, conf=0.2)
    car_counts = [0, 0, 0, 0]; ped_counts = [0, 0, 0, 0]
    
    if results[0].boxes:
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0]); center_x = (x1 + x2) // 2; center_y = (y1 + y2) // 2

            if label in VEHICLE_LABELS:
                for i, (x_min, y_min, x_max, y_max) in CAR_ROIS.items():
                    if x_min < center_x < x_max and y_min < center_y < y_max: car_counts[i] += 1; cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2); break
            
            elif label == PEDESTRIAN_LABEL:
                for i, (x_min, y_min, x_max, y_max) in PEDESTRIAN_ROIS.items():
                    if x_min < center_x < x_max and y_min < center_y < y_max: ped_counts[i] += 1; cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2); break
    
    return car_counts, sum(ped_counts)

# --- TRACKING ARDUINO STATE ---
# We use this flag to know when the Arduino is busy running the Yellow/Red cycle
ARDUINO_IS_BUSY = False 
# ---

while True:
    frames = {}
    
    for cam_index in caps:
        success, frame = caps[cam_index].read()
        if not success: frame = black_frame 
        else: frame = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))
        frames[cam_index] = frame

    frame_0 = frames.get(0, black_frame); frame_1 = frames.get(1, black_frame)
    frame_2 = frames.get(2, black_frame); frame_3 = frames.get(3, black_frame)
    row1 = np.hstack((frame_0, frame_1)); row2 = np.hstack((frame_2, frame_3))
    combined_frame = np.vstack((row1, row2))

    car_counts, total_pedestrians = get_counts_and_process_frame(combined_frame)

    # --- ACTION DECISION LOGIC (Peds have priority, then cars) ---
    action = current_green_light # Default: Keep the current light (0)
    
    # Check if Arduino is done with the last cycle
    if ARDUINO_IS_BUSY and wait_for_arduino_ack(arduino):
        print("--- ARDUINO CYCLE COMPLETED. READY FOR NEW COMMAND. ---")
        ARDUINO_IS_BUSY = False
        # After Peds (P) finish, we reset the action to find the busiest car lane
        last_action_sent = 'C' 

    if not ARDUINO_IS_BUSY:
        if total_pedestrians >= PEDESTRIAN_THRESHOLD:
            desired_action_char = 'P' # Pedestrian Walk (Highest Priority)
        else:
            # If Peds are gone, find the busiest car lane (or keep current if no cars)
            if sum(car_counts) == 0:
                desired_action_char = str(current_green_light) 
            else:
                desired_action_char = str(np.argmax(car_counts)) # Busiest car lane (0-3)

        # Send command if needed
        if desired_action_char != last_action_sent:
            last_action_sent = desired_action_char
            
            # Update the integer tracker for car logic display
            current_green_light = int(last_action_sent) if last_action_sent != 'P' else 4
            
            print(f"--- NEW ACTION: {last_action_sent} (Peds: {total_pedestrians}) ---")
            
            # --- SEND TO ARDUINO ---
            if arduino is not None:
                arduino.reset_input_buffer() 
                arduino.write(last_action_sent.encode())
                
                # If we sent a PEDESTRIAN trigger, the Arduino is now BUSY for 2 seconds
                if last_action_sent == 'P':
                    ARDUINO_IS_BUSY = True
                    play_voice_command(VOICE_FILE_WALK)

    # --- DISPLAY THE FRAME & INFO ---
    for i, (x1, y1, x2, y2) in CAR_ROIS.items(): cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    font_scale = SCREEN_HEIGHT / 720.0 
    status_text = f"ACTION: {last_action_sent} | Peds: {total_pedestrians} | Cars: {car_counts}"
    cv2.putText(combined_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, (0, 255, 255), 3)
    
    cv2.imshow(WINDOW_NAME, combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- CLEANUP ---
print("Stopping demo...")
for cap in caps.values(): cap.release()
cv2.destroyAllWindows()
if arduino is not None: arduino.close()
pygame.mixer.quit()