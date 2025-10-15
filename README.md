# WeSIGN Backend - Indian Sign Language Recognition API

Flask-based REST API for real-time Indian Sign Language (ISL) gesture recognition using LSTM model and MediaPipe.

## 🚀 Features

- **Real-time Gesture Recognition** - Process video frames and detect ISL gestures
- **LSTM Model** - Deep learning model for sequence-based sign recognition
- **MediaPipe Integration** - Accurate hand and pose landmark detection
- **REST API** - Easy integration with any frontend
- **Session Management** - Multiple concurrent users supported
- **Confidence Scores** - Get prediction confidence for each gesture

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for testing)
- `sign_lstm_model.keras` - Trained LSTM model file
- `sign_encoder.pkl` - Label encoder file

## 🛠️ Installation

1. **Navigate to backend directory:**
   ```bash
   cd "d:/Niharika Maurya-D/COLLEGE/WESIGN/words_wesign-main"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Start the Backend Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Test with Standalone Script

To test the model directly with your webcam:

```bash
python run_inference.py
```

Press 'q' to quit.

## 📡 API Endpoints

### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoder_loaded": true
}
```

### 2. Predict Gesture
```http
POST /api/predict
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "session_id": "session_12345"
}
```

**Response (collecting frames):**
```json
{
  "success": true,
  "gesture": null,
  "confidence": 0,
  "buffer_length": 15,
  "message": "Collecting frames... 15/40"
}
```

**Response (gesture detected):**
```json
{
  "success": true,
  "gesture": "Hello",
  "confidence": 87.5,
  "buffer_length": 40
}
```

### 3. Reset Session
```http
POST /api/reset
```

**Request Body:**
```json
{
  "session_id": "session_12345"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Session reset successfully"
}
```

## 🔧 How It Works

1. **Frame Capture** - Frontend captures video frames from webcam
2. **Send to Backend** - Frames sent as base64 encoded images
3. **MediaPipe Processing** - Extract hand and pose landmarks
4. **Normalization** - Normalize landmarks relative to shoulder position
5. **Sequence Buffer** - Collect 40 frames of landmarks
6. **LSTM Prediction** - Model predicts gesture from sequence
7. **Return Result** - Send gesture and confidence back to frontend

## 📊 Model Details

- **Architecture:** LSTM (Long Short-Term Memory)
- **Input:** 40 frames × 189 features (pose + hand landmarks)
- **Output:** Gesture class + confidence score
- **Framework:** TensorFlow/Keras

### Landmark Features (189 total)
- **Pose landmarks:** 21 points × 3 coordinates = 63 features
- **Left hand landmarks:** 21 points × 3 coordinates = 63 features
- **Right hand landmarks:** 21 points × 3 coordinates = 63 features

## 🔄 Integration with Frontend

The frontend (`wesign-frontend`) connects to this backend via REST API:

1. Frontend captures video frames
2. Sends frames to `/api/predict` endpoint
3. Backend processes and returns predictions
4. Frontend displays gesture and confidence

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `sign_lstm_model.keras` exists in the directory
- Check TensorFlow version compatibility

### Camera Issues
- Grant camera permissions in browser
- Check if camera is being used by another application

### CORS Errors
- Backend has CORS enabled for all origins
- Check if backend is running on port 5000

### Low Confidence Scores
- Ensure good lighting conditions
- Position hands clearly in frame
- Make deliberate, clear gestures

## 📁 File Structure

```
words_wesign-main/
├── app.py                    # Flask API server
├── run_inference.py          # Standalone testing script
├── sign_lstm_model.keras     # Trained LSTM model
├── sign_encoder.pkl          # Label encoder
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔐 Security Notes

- This is a development server - use production WSGI server for deployment
- Consider adding authentication for production use
- Implement rate limiting for API endpoints

## 📝 License

MIT License

## 🙏 Acknowledgments

- MediaPipe by Google
- TensorFlow/Keras team
- Indian Sign Language community
"# wesign-fb" 
