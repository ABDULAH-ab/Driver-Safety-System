from flask import Flask, render_template, Response, jsonify, request
from driver_safety_system import DriverSafetySystem
import threading
import time
import cv2
import numpy as np
import logging

app = Flask(__name__)
safety_system = DriverSafetySystem(web_mode=True)
# system_thread = None
is_streaming = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        logging.error("No frame provided in request")
        return jsonify({'error': 'No frame provided'}), 400
    
    frame_file = request.files['frame']
    # Read the image file into a numpy array
    nparr = np.frombuffer(frame_file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        logging.error("Invalid frame data received")
        return jsonify({'error': 'Invalid frame data'}), 400
    
    # Start the safety system if it's not running
    if not safety_system.running:
        logging.info("Starting safety system")
        safety_system.start()
    
    # Process the frame
    safety_system.process_frame(frame)
    
    # Wait a short time for processing to complete
    time.sleep(0.05)
    
    return jsonify({'status': 'success'}), 200

@app.route('/get_processed_frame')
def get_processed_frame():
    if not safety_system.running:
        return '', 204
        
    frame = safety_system.get_latest_frame()
    if frame is None:
        return '', 204
    
    return Response(frame, mimetype='image/jpeg')

@app.route('/toggle_stream')
def toggle_stream():
    global is_streaming
    if is_streaming:
        logging.info("Stopping stream")
        is_streaming = False
        safety_system.stop()
    else:
        logging.info("Starting stream")
        is_streaming = True
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    def generate():
        while safety_system.running:
            frame = safety_system.get_latest_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
