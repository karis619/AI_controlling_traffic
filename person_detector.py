# This file is: person_detector.py

import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import pygame  # Used for playing the audio

# --- 1. CONFIGURATION ---

# Path to your optimized model
MODEL_PATH = "vision/models/yolov8n_openvino_model/"

# Voice configuration
VOICE_MESSAGE = "A person has been detected."
VOICE_FILE = "person_detected.mp3"
VOICE_LANG = 'en'

# --- 2. ONE-TIME AUDIO SETUP ---

# Initialize the pygame mixer for audio
try:
    pygame.mixer.init()
    print("Pygame mixer initialized successfully.")
except pygame.error as e:
    print(f"Error initializing pygame mixer: {e}")
    print("Please ensure you have a working audio output device.")
    sys.exit()

# Create the voice file if it doesn't exist
if not os.path.exists(VOICE_FILE):
    print(f"Creating voice file: '{VOICE_FILE}'...")
    try:
        tts = gTTS(text=VOICE_MESSAGE, lang=VOICE_LANG, slow=False)
        tts.save(VOICE_FILE)
        print("Voice file created successfully.")
    except Exception as e:
        print(f"Error creating voice file: {e}")
        print("Please check your internet connection.")
        sys.exit()
else:
    print(f"Voice file '{VOICE_FILE}' already exists.")

# --- 3. ONE-TIME MODEL & CAMERA SETUP ---

# Load your CPU-optimized OpenVINO model
try:
    model = YOLO(MODEL_PATH)
    print("Successfully loaded OpenVINO model.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}")
    print("Make sure the model exists at this location.")
    print(e)
    sys.exit()

# Start the webcam feed (0 is usually the default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("--- Starting Person Detector ---")
print("Press 'q' in the window to quit.")

# --- 4. MAIN DETECTION LOOP ---

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Error: Failed to grab frame from webcam.")
        break

    # --- Run YOLO detection ---
    # We run detection on every frame.
    results = model.predict(frame, verbose=False)

    person_was_detected = False

    # Process the results
    for box in results[0].boxes:
        # Get the class name
        label = model.names[int(box.cls[0])]

        if label == "person":
            person_was_detected = True
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw a rectangle and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box
            cv2.putText(frame, "Person", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Since we only care if *a* person is found, we can stop looping
            break 

    # --- 5. TRIGGER VOICE (with cooldown) ---
    
    # Check if a person was found AND if the audio is not already playing
    if person_was_detected and not pygame.mixer.music.get_busy():
        print("Person detected! Playing voice message...")
        pygame.mixer.music.load(VOICE_FILE)
        pygame.mixer.music.play()

    # --- 6. DISPLAY THE FRAME ---
    cv2.imshow("Person Detector (Press 'q' to quit)", frame)

    # --- 7. QUIT CONDITION ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. CLEANUP ---
print("Stopping detector...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
sys.exit()