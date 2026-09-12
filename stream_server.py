from flask import Flask, Response
import cv2
import sys

app = Flask(__name__)

# --- CONFIGURATION ---
# !!! CRITICAL: CHANGE THIS FOR EACH LAPTOP !!!
# Laptop 1: PORT = 8000
# Laptop 2: PORT = 8001
# Laptop 3: PORT = 8002
# Laptop 4: PORT = 8003
PORT_NUMBER = 8000 
# ---

# Initialize the camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    # Set a standard resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
except Exception as e:
    print(f"Error initializing webcam: {e}")
    print("Is it being used by another program (like Zoom or Teams)?")
    sys.exit()

def gen_frames():
    """A generator function that yields camera frames as JPEG images."""
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera. Retrying...")
            cap.release()
            cap.open(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            continue
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the special multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """The main route that streams the video."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"--- Starting webcam server on port {PORT_NUMBER} ---")
    print(f"Find your IP with 'ipconfig' in cmd.")
    print(f"Your stream URL is: http://<your_ip>:{PORT_NUMBER}/video_feed")
    # 'host=0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=PORT_NUMBER, threaded=True)