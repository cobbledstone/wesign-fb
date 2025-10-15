from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ----------------------------
# Load model + label encoder
# ----------------------------
print("Loading model...")
model = load_model("sign_lstm_model.keras")
encoder = joblib.load("sign_encoder.pkl")
print("Model loaded successfully!")

# ----------------------------
# Mediapipe setup
# ----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Global variables for each session
# ----------------------------
sequence_length = 40
user_buffers = {}  # Store buffer for each user session

# ----------------------------
# Normalization function
# ----------------------------
def extract_normalized_landmarks(results):
    num_landmarks = (21 + 21 + 21) * 3  # 189
    
    if not results.pose_landmarks:
        return np.zeros(num_landmarks)

    pose = results.pose_landmarks.landmark
    left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

    # Select first 21 pose landmarks
    pose_idx = list(range(21))
    pose_points = np.array([[pose[i].x, pose[i].y, pose[i].z] for i in pose_idx])

    if len(left_hand) == 21:
        left_points = np.array([[lm.x, lm.y, lm.z] for lm in left_hand])
    else:
        left_points = np.zeros((21, 3))

    if len(right_hand) == 21:
        right_points = np.array([[lm.x, lm.y, lm.z] for lm in right_hand])
    else:
        right_points = np.zeros((21, 3))

    # Shoulder midpoint
    left_shoulder = np.array([pose[11].x, pose[11].y, pose[11].z])
    right_shoulder = np.array([pose[12].x, pose[12].y, pose[12].z])
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    # Normalize
    pose_points -= shoulder_mid
    left_points -= shoulder_mid
    right_points -= shoulder_mid

    # Scale normalization (shoulder distance)
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist > 0:
        pose_points /= shoulder_dist
        left_points /= shoulder_dist
        right_points /= shoulder_dist

    all_points = np.concatenate([pose_points, left_points, right_points], axis=0).flatten()
    return all_points

# ----------------------------
# REST API Routes
# ----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': encoder is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_frame():
    """Process a single frame and return prediction"""
    try:
        data = request.json
        image_data = data.get('image')
        session_id = data.get('session_id', 'default')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize buffer for this session if not exists
        if session_id not in user_buffers:
            user_buffers[session_id] = deque(maxlen=sequence_length)
        
        buffer = user_buffers[session_id]
        
        # Process with MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Extract landmarks
            keypoints = extract_normalized_landmarks(results)
            buffer.append(keypoints)
            
            # Predict only when we have full sequence
            if len(buffer) == sequence_length:
                X_input = np.expand_dims(buffer, axis=0)
                preds = model.predict(X_input, verbose=0)
                pred_class = encoder.inverse_transform([np.argmax(preds)])
                confidence = float(np.max(preds))
                
                return jsonify({
                    'success': True,
                    'gesture': pred_class[0],
                    'confidence': confidence * 100,
                    'buffer_length': len(buffer)
                })
            else:
                return jsonify({
                    'success': True,
                    'gesture': None,
                    'confidence': 0,
                    'buffer_length': len(buffer),
                    'message': f'Collecting frames... {len(buffer)}/{sequence_length}'
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the buffer for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in user_buffers:
            user_buffers[session_id].clear()
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ----------------------------
# WebSocket Events (Alternative real-time approach)
# ----------------------------
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to WeSIGN backend'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    # Clean up buffer
    if request.sid in user_buffers:
        del user_buffers[request.sid]

@socketio.on('process_frame')
def handle_frame(data):
    """Process frame via WebSocket"""
    try:
        session_id = request.sid
        image_data = data.get('image')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize buffer for this session
        if session_id not in user_buffers:
            user_buffers[session_id] = deque(maxlen=sequence_length)
        
        buffer = user_buffers[session_id]
        
        # Process with MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Extract landmarks
            keypoints = extract_normalized_landmarks(results)
            buffer.append(keypoints)
            
            # Predict when buffer is full
            if len(buffer) == sequence_length:
                X_input = np.expand_dims(buffer, axis=0)
                preds = model.predict(X_input, verbose=0)
                pred_class = encoder.inverse_transform([np.argmax(preds)])
                confidence = float(np.max(preds))
                
                emit('prediction', {
                    'gesture': pred_class[0],
                    'confidence': confidence * 100,
                    'buffer_length': len(buffer)
                })
            else:
                emit('status', {
                    'buffer_length': len(buffer),
                    'message': f'Collecting frames... {len(buffer)}/{sequence_length}'
                })
                
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('reset_buffer')
def handle_reset():
    """Reset buffer via WebSocket"""
    session_id = request.sid
    if session_id in user_buffers:
        user_buffers[session_id].clear()
    emit('reset_complete', {'message': 'Buffer reset'})

if __name__ == '__main__':
    print("Starting WeSIGN Backend Server...")
    print("Server running on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
