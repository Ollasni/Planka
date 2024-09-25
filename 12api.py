from fastapi import FastAPI, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
import io
from starlette.responses import StreamingResponse

app = FastAPI()

# Load models and scalers
with open("./model/LR_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

with open("./model/input_scaler.pkl", "rb") as f2:
    input_scaler = pickle.load(f2)

# Mediapipe helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", 
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", 
    "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", 
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def extract_important_keypoints(results) -> list:
    if not results or not results.pose_landmarks:
        return [0] * len(HEADERS[1:])  # Return zeros if pose landmarks not found
    
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

# Определяем порог вероятности для классификации
prediction_probability_threshold = 0.6  # Глобальная переменная

def classify_pose(image: np.ndarray):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process pose landmarks
    results = pose_model.process(image_rgb)
    
    # Extract keypoints
    if results.pose_landmarks:
        row = extract_important_keypoints(results)
        X = pd.DataFrame([row], columns=HEADERS[1:])
        X_scaled = input_scaler.transform(X)
        
        # Predict pose class
        predicted_class = sklearn_model.predict(X_scaled)[0]
        predicted_class = get_class(predicted_class)
        prediction_probability = sklearn_model.predict_proba(X_scaled)[0]

        # Determine the feedback based on the classification result
        if prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
            if predicted_class == "C":
                feedback = "Correct"
            elif predicted_class == "L":
                feedback = "Low back"
            elif predicted_class == "H":
                feedback = "High back"
            else:
                feedback = "Unknown"
        else:
            feedback = "Unknown"

        return {
            "class": predicted_class,
            "probability": prediction_probability.tolist(),
            "feedback": feedback,
            "results": results
        }
    
    return {
        "class": "unknown",
        "probability": [],
        "feedback": "Unknown",
        "results": None
    }

def get_class(prediction: float) -> str:
    return {0: "C", 1: "H", 2: "L"}.get(prediction, "unknown")

# Function to generate frames from camera
def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Classify pose for the current frame
            classification = classify_pose(frame)

            # If pose is detected, draw the landmarks and feedback
            if classification["results"]:
                mp_drawing.draw_landmarks(
                    frame, 
                    classification["results"].pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )
            
            # Display the feedback text on the frame
            feedback_text = classification["feedback"]
            cv2.putText(frame, feedback_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Return the frame in a streaming response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video from the camera
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
