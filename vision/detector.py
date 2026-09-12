# This file is: /vision/detector.py

import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = "vision/models/yolov8n_openvino_model/"
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded OpenVINO model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}.")
    print("Make sure you have run 'yolo export model=yolov8n.pt format=openvino'")
    print(e)
    exit()

VEHICLE_LABELS = {"car", "bus", "truck", "rickshaw", "bike", "bicycle", "motorcycle"}

# --- 2. DEFINE YOUR REGIONS OF INTEREST (ROIs) ---
#
# !!! CRITICAL: MUST-CHANGE !!!
# These pixel coordinates MUST match the waiting zones in your Pygame window.
# Format: [x_min, y_min, x_max, y_max]
# Directions: 0='right', 1='down', 2='left', 3='up'
#
ROIS = {
    0: [0, 250, 580, 500],    # ROI for 'right' lane (index 0)
    1: [600, 0, 850, 320],    # ROI for 'down' lane (index 1)
    2: [810, 350, 1400, 600], # ROI for 'left' lane (index 2)
    3: [500, 545, 750, 800]   # ROI for 'up' lane (index 3)
}

# --- 3. THE "EYES" FUNCTION ---

def get_state_from_vision(frame: np.ndarray) -> np.ndarray:
    """
    Takes a Pygame frame (as a numpy array) and returns the
    vehicle counts for each of the 4 lanes.
    """
    lane_counts = [0, 0, 0, 0]

    results = model.predict(frame, verbose=False)

    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]

        if label in VEHICLE_LABELS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            for direction_index, (x_min, y_min, x_max, y_max) in ROIS.items():
                if x_min < center_x < x_max and y_min < center_y < y_max:
                    lane_counts[direction_index] += 1
                    break 

    return np.array(lane_counts, dtype=np.int32)