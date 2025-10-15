import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from tensorflow.keras.models import load_model

# ----------------------------
# Load model + label encoder
# ----------------------------
model = load_model("sign_lstm_model.keras")

encoder = joblib.load("sign_encoder.pkl")


# ----------------------------
# Mediapipe setup
# ----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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
# Real-time inference loop
# ----------------------------
sequence_length = 40  # must match training
buffer = deque(maxlen=sequence_length)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe inference
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract & store
        keypoints = extract_normalized_landmarks(results)
        buffer.append(keypoints)

        # Predict only when we have full sequence
        if len(buffer) == sequence_length:
            X_input = np.expand_dims(buffer, axis=0)  # shape (1,40,189)
            preds = model.predict(X_input, verbose=0)
            pred_class = encoder.inverse_transform([np.argmax(preds)])
            confidence = np.max(preds)

            # Show prediction on screen
            if (True):
                cv2.putText(image, f"{pred_class[0]} ({confidence:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2, cv2.LINE_AA)
            

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Sign Recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
