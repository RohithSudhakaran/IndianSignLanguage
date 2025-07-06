import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import deque, Counter
import time
import os
import sys

# 🖐 Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# 📂 Load CNN model and label encoder
try:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'hand_gesture_cnn_model.h5')
    encoder_path = os.path.join(base_dir, '..', 'models', 'label_encoder.pkl')

    cnn_model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)

    print("[✅] CNN model and label encoder loaded successfully.")
except Exception as e:
    print(f"[❌] Failed to load model or encoder: {e}")
    sys.exit(1)

# 🎥 Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌] Could not access webcam.")
    exit()

print("Press 'q' to quit.")

# 📝 Prediction smoothing
prediction_history = deque(maxlen=10)
stable_prediction = None
predicted_sequence = []

# ⚡ Flash variables
capture_flash_frames = 0
flash_duration = 5

# 🐢 Prediction delay
last_prediction_time = 0
prediction_delay = 1.5

# 📜 One-hand gestures
one_hand_numbers = set([str(i) for i in range(1, 10)])
one_hand_letters = set(['C', 'I', 'L', 'O', 'U', 'V'])

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    num_hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if results.multi_hand_landmarks:
        first_hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[pt.x * w, pt.y * h] for pt in first_hand_landmarks.landmark])

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        flattened = landmarks.flatten().tolist()
        normalized = np.array(flattened) / np.max(flattened)

        prediction_probs = cnn_model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        if (
            (predicted_label in one_hand_numbers or predicted_label in one_hand_letters and num_hands_detected == 1)
            or (predicted_label not in one_hand_numbers and predicted_label not in one_hand_letters and num_hands_detected == 2)
        ):
            if confidence >= 0.90:
                prediction_history.append(predicted_label)
                most_common, count = Counter(prediction_history).most_common(1)[0]
                current_time = time.time()

                if (
                    most_common != stable_prediction
                    and count >= 8
                    and current_time - last_prediction_time >= prediction_delay
                ):
                    stable_prediction = most_common
                    predicted_sequence.append(stable_prediction)
                    print(f"[✅] CONFIRMED: {stable_prediction} (Confidence: {confidence:.2f})")
                    capture_flash_frames = flash_duration
                    last_prediction_time = current_time
            else:
                prediction_history.clear()
        else:
            prediction_history.clear()
            stable_prediction = None
            capture_flash_frames = 0

        all_landmarks = np.vstack(
            [[pt.x * w, pt.y * h] for hand in results.multi_hand_landmarks for pt in hand.landmark]
        )
        x_min = int(np.min(all_landmarks[:, 0])) - 20
        y_min = int(np.min(all_landmarks[:, 1])) - 20
        x_max = int(np.max(all_landmarks[:, 0])) + 20
        y_max = int(np.max(all_landmarks[:, 1])) + 20

        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, w), min(y_max, h)

        box_color = (255, 255, 255) if capture_flash_frames > 0 else (0, 255, 0)
        capture_flash_frames = max(0, capture_flash_frames - 1)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 3)
        cv2.putText(image, f'{stable_prediction if stable_prediction else "..."}',
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    else:
        prediction_history.clear()
        stable_prediction = None
        capture_flash_frames = 0
        cv2.putText(image, 'No hand detected', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    sequence_text = " ".join(predicted_sequence[-10:])
    cv2.rectangle(image, (0, h - 60), (w, h), (50, 50, 50), -1)
    cv2.putText(image, sequence_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('🖐 Hand Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
