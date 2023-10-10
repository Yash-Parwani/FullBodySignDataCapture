import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load the RNN model
model = keras.models.load_model('rnn_model.h5')  # Replace with the path to your RNN model

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            pose = results.pose_landmarks.landmark
            pose_row = [(landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in pose]

            face = results.face_landmarks.landmark
            face_row = [(landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in face]

            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = [(landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in right_hand]

            left_hand = results.left_hand_landmarks.landmark
            left_hand_row = [(landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in left_hand]

            row = pose_row + face_row + left_hand_row + right_hand_row

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)
            body_language_prob = model.predict_proba(X)

            # Convert the class to a string label
            class_labels = ['Hello', 'ThankYou', 'ILoveYou']  # Modify these labels as needed
            detected_label = class_labels[np.argmax(body_language_class)]

            print(detected_label, body_language_prob)

            # Display the detected label on the video feed
            coords = tuple(np.multiply(
                np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                [640, 480]).astype(int))

            cv2.rectangle(image,
                          (coords[0], coords[1] + 5),
                          (coords[0] + len(detected_label) * 20, coords[1] - 30),
                          (245, 117, 16), -1)
            cv2.putText(image, detected_label,
                        coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Raw Webcam Feed', image)

        except:
            pass

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
