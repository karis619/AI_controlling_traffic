import cv2
import sys
import os
from ultralytics import YOLO

# --- 1. CONFIGURATION ---

# Path to your optimized OpenVINO model
MODEL_PATH = "vision/models/yolov8n_openvino_model/"

# !!! --- SET THESE FOLDERS --- !!!
# The folder where you will put your toy pictures
IMAGE_DIR = "test_images" 
# The folder where the script will save the results
OUTPUT_DIR = "output_images"

# List of objects we care about
LABELS_TO_DETECT = {"car", "bus", "truck", "motorcycle", "person"}

# --- 2. SETUP ---

# Load the YOLO "Eyes"
print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Successfully loaded OpenVINO model.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Check if the test image folder exists
if not os.path.exists(IMAGE_DIR):
    print(f"Error: The folder '{IMAGE_DIR}' was not found.")
    print(f"Please create it in your project directory and add your toy images.")
    sys.exit()

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Test images will be read from: {IMAGE_DIR}")
print(f"Detected images will be saved to: {OUTPUT_DIR}")
print("--- Starting Batch Detection ---")

# --- 3. MAIN DETECTION LOOP ---

# Loop through every file in the IMAGE_DIR
for filename in os.listdir(IMAGE_DIR):
    # Check if it's an image file
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        # Build the full file paths
        image_path = os.path.join(IMAGE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")

        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read {filename}, skipping.")
            continue

        print(f"\nProcessing {filename}...")

        # Run detection
        results = model.predict(frame, verbose=False)

        detection_count = 0

        # Loop through all detected objects for this image
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            
            if label in LABELS_TO_DETECT:
                detection_count += 1
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw the box and label
                color = (0, 0, 255) if label == "person" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ({confidence*100:.0f}%)"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"  > Found: {text}")

        # Save the new image (with boxes) to the output folder
        cv2.imwrite(output_path, frame)
        print(f"  > Saved result to {output_path}")

print("\n--- Batch processing complete. ---")