import cv2
import numpy as np
import sys

# --- 1. CONFIGURATION ---

# !!! CRITICAL: PASTE YOUR LAPTOP IPs AND PORTS HERE !!!
#
# This is the *only* part you need to edit.
# You can list 1, 2, 3, or all 4. The script will handle it.
#
URLS = {
    0: "http://10.224.246.99:8001/video_feed",  # Top-Left (Camera 0)
    #1: "http://192.168.2.103:8001/video_feed",  # This is Cam 1 (Top-Right)
    #2: "http://192.168.2.100:8002/video_feed"   # This is Cam 2 (Bottom-Left) # Top-Right (Camera 1)
    
    # --- Add your other cameras here when they are ready ---
    # 2: "http://<IP_FOR_CAM_2>:<PORT_FOR_CAM_2>/video_feed", # Bottom-Left (Camera 2)
    # 3: "http://<IP_FOR_CAM_3>:<PORT_FOR_CAM_3>/video_feed"  # Bottom-Right (Camera 3)
}

# We will resize each stream to this standard size
TILE_WIDTH = 480
TILE_HEIGHT = 360

# --- 2. CONNECT TO CAMERAS ---
caps = {}
for (cam_index, url) in URLS.items():
    print(f"Attempting to connect to camera {cam_index} at {url}...")
    caps[cam_index] = cv2.VideoCapture(url)
    
    if not caps[cam_index].isOpened():
        print(f"Error: Could not open stream for camera {cam_index} at {url}")
        print(f"!!! Camera {cam_index} will be marked as OFFLINE. Check firewall/IP. !!!")
        # We don't exit, just remove it from the list of active caps
        del caps[cam_index] 

print(f"\nSuccessfully opened {len(caps)} camera stream(s).")
print("Press 'q' in the window to quit.")

# --- 3. CREATE A BLANK "OFFLINE" FRAME ---
# This is our placeholder for any camera that is not connected
black_frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
cv2.putText(black_frame, "OFFLINE", (TILE_WIDTH // 2 - 70, TILE_HEIGHT // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# --- 4. MAIN DISPLAY LOOP ---
while True:
    frames = {}
    
    # Read one frame from each *successfully connected* camera
    for cam_index in caps:
        success, frame = caps[cam_index].read()
        
        if not success:
            # If the stream fails mid-run, show the black frame
            frame = black_frame 
        else:
            # If successful, resize it
            frame = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))
        
        frames[cam_index] = frame

    # --- 5. DYNAMICALLY BUILD THE 2x2 GRID ---
    #
    # This is the robust fix.
    # It will get the frame from the dictionary if it exists.
    # If not (e.g., you only listed 2 URLs), it will use the 'black_frame'.
    # This WILL NOT crash with an IndexError or KeyError.
    
    frame_0 = frames.get(0, black_frame) # Get Cam 0 or default
    frame_1 = frames.get(1, black_frame) # Get Cam 1 or default
    frame_2 = frames.get(2, black_frame) # Get Cam 2 or default
    frame_3 = frames.get(3, black_frame) # Get Cam 3 or default

    # Stitch the top row together
    row1 = np.hstack((frame_0, frame_1))
    # Stitch the bottom row together
    row2 = np.hstack((frame_2, frame_3))
    # Stack the two rows vertically
    combined_frame = np.vstack((row1, row2))

    # --- 6. DISPLAY THE COMBINED FRAME ---
    cv2.imshow("4-Camera Tiled Feed (Press 'q' to quit)", combined_frame)

    # --- 7. QUIT CONDITION ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. CLEANUP ---
print("Stopping streams...")
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()