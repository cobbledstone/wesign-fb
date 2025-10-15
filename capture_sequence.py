import cv2
import numpy as np
import mediapipe as mp
import os
from time import sleep


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "dataset"
classes = ["Thank_You"]   # add all your classes
sequences_per_class = 50
frames_per_seq = 40

import numpy as np

def extract_normalized_landmarks(results):
    """
    Extracts pose + left hand + right hand landmarks,
    normalizes by shoulder midpoint, scales by shoulder distance,
    returns flat np.array of length 189.
    If missing data, returns zeros.
    """
    num_landmarks = (21 + 21 + 21) * 3  # = 189
    
    # Check if pose landmarks exist
    if not results.pose_landmarks:
        return np.zeros(num_landmarks)
    
    pose = results.pose_landmarks.landmark
    left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

    # Select 21 pose points
    pose_idx = list(range(21))
    pose_points = np.array([[pose[i].x, pose[i].y, pose[i].z] for i in pose_idx])

    # Handle missing hands
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

    # Normalize by subtracting shoulder midpoint
    pose_points -= shoulder_mid
    left_points -= shoulder_mid
    right_points -= shoulder_mid

    # Scale normalization (shoulder distance)
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist > 0:
        pose_points /= shoulder_dist
        left_points /= shoulder_dist
        right_points /= shoulder_dist

    # Flatten
    all_points = np.concatenate([pose_points, left_points, right_points], axis=0).flatten()
    return all_points



mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def draw_styled_landmarks(image, results):
    # Draw face connections
    '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )'''
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                             ) 

def extract_keypoints(results):
    # Left hand (21 landmarks)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    # Right hand
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    # Pose (first 21)
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:21]]) if results.pose_landmarks else np.zeros((21,3))
    
    return np.concatenate([lh, rh, pose]).flatten()  # (189,)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for gesture in classes:
        gesture_dir = os.path.join(DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        for seq in range(50,100,1):
            sequence = []

            print(f"Recording {gesture} - Sequence {seq+1}/{sequences_per_class}")
            sleep(1)
            print(3)
            sleep(1)
            print(2)
            sleep(1)
            print(1)
            sleep(1)
            print("go")
            for frame_num in range(frames_per_seq):
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  
                results = holistic.process(image)
                
                image.flags.writeable = True
                # extract landmarks
                keypoints = extract_normalized_landmarks(results)
                sequence.append(keypoints)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # show preview
                draw_styled_landmarks(image,results)
                
                cv2.imshow("Recording", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            np.save(os.path.join(gesture_dir, f"sample{seq}.npy"), np.array(sequence))

cap.release()
cv2.destroyAllWindows()
