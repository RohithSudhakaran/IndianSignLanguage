import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import deque, Counter

# 🖐 Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2  # Detect up to 2 hands
)

# 📂 Load CNN model and label encoder
cnn_model = tf.keras.models.load_model("hand_gesture_cnn_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
print("[✅] CNN model and label encoder loaded successfully.")

# 🎥 Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌] Could not access webcam.")
    exit()

print("Press 'q' to quit.")

# 📝 Prediction smoothing
prediction_history = deque(maxlen=10)  # Store last 10 predictions
stable_prediction = None
predicted_sequence = []  # Store all confirmed gestures

# ⚡ Capture flash variables
capture_flash_frames = 0
flash_duration = 5  # How many frames the flash lasts

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # ✅ Get image dimensions at the top
    h, w, _ = image.shape

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Initialize list to collect all landmarks from all hands
    all_landmarks = []

    # 🖐 Detect hands and collect landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark])
            all_landmarks.extend(landmarks.tolist())  # Add landmarks to combined list

            # ✅ Draw landmarks for each hand
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # ✅ Calculate one bounding box around all hands
        all_landmarks_np = np.array(all_landmarks)
        x_min = int(np.min(all_landmarks_np[:, 0])) - 20
        y_min = int(np.min(all_landmarks_np[:, 1])) - 20
        x_max = int(np.max(all_landmarks_np[:, 0])) + 20
        y_max = int(np.max(all_landmarks_np[:, 1])) + 20

        # Ensure box stays within frame
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, w), min(y_max, h)

        # ✅ Use only the first hand for prediction
        first_hand_landmarks = results.multi_hand_landmarks[0]
        first_hand_coords = np.array(
            [[point.x * w, point.y * h] for point in first_hand_landmarks.landmark]
        )
        flattened_landmarks = first_hand_coords.flatten().tolist()
        normalized_landmarks = np.array(flattened_landmarks) / np.max(flattened_landmarks)

        # Predict gesture with CNN
        prediction = cnn_model.predict(
            np.expand_dims(normalized_landmarks, axis=0),
            verbose=0
        )
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        # Add to prediction history
        prediction_history.append(predicted_label)

        # Find most common prediction in history
        most_common_prediction, count = Counter(prediction_history).most_common(1)[0]

        # Confirm prediction
        if most_common_prediction != stable_prediction and count > 5:  # Threshold = 5 of last 10 frames
            stable_prediction = most_common_prediction
            predicted_sequence.append(stable_prediction)  # Add to sequence
            print(f"[✅] Predicted: {stable_prediction}")
            capture_flash_frames = flash_duration  # Trigger capture flash

        # 🖥 Draw combined bounding box
        if capture_flash_frames > 0:
            box_color = (255, 255, 255)  # Flash white (capture effect)
            capture_flash_frames -= 1
        else:
            box_color = (0, 255, 0)  # Normal green box

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 3)
        cv2.putText(image, f'{stable_prediction if stable_prediction else "..."}',
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    else:
        # If no hand detected, clear prediction history
        prediction_history.clear()
        stable_prediction = None
        capture_flash_frames = 0  # Stop flash
        cv2.putText(image, 'No hand detected', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 📝 Display predicted sequence below webcam feed
    sequence_text = " ".join(predicted_sequence[-10:])  # Show last 10 gestures
    cv2.rectangle(image, (0, h - 60), (w, h), (50, 50, 50), -1)  # Background bar
    cv2.putText(image, sequence_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show webcam feed
    cv2.imshow('🖐 Hand Gesture Recognition', image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
