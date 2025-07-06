import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import deque, Counter
import time

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

# 🐢 Prediction delay variables
last_prediction_time = 0
prediction_delay = 1.5  # seconds to wait before next prediction

# 📜 One-hand gestures
one_hand_numbers = set([str(i) for i in range(1, 10)])  # '1' to '9'
one_hand_letters = set(['C', 'I', 'L', 'O', 'U', 'V'])  # Added 'O' here

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # ✅ Get image dimensions
    h, w, _ = image.shape

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 🖐 Count detected hands
    num_hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    # 🖐 Detect hands and collect landmarks
    if results.multi_hand_landmarks:
        # ✅ Use only the first hand for prediction
        first_hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[point.x * w, point.y * h] for point in first_hand_landmarks.landmark])

        # ✅ Draw landmarks for all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Flatten and normalize landmarks
        flattened_landmarks = landmarks.flatten().tolist()
        normalized_landmarks = np.array(flattened_landmarks) / np.max(flattened_landmarks)

        # ✅ Predict gesture with CNN
        prediction_probs = cnn_model.predict(
            np.expand_dims(normalized_landmarks, axis=0),
            verbose=0
        )[0]
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]  # Get confidence score
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        # ✅ Check gesture filtering based on hand count
        if (
            (predicted_label in one_hand_numbers or predicted_label in one_hand_letters and num_hands_detected == 1)
            or (predicted_label not in one_hand_numbers and predicted_label not in one_hand_letters and num_hands_detected == 2)
        ):
            # ✅ Only consider prediction if confidence is high enough
            confidence_threshold = 0.90  # Accept predictions >90%
            if confidence >= confidence_threshold:
                prediction_history.append(predicted_label)

                # ✅ Find most common prediction in history
                most_common_prediction, count = Counter(prediction_history).most_common(1)[0]

                # ✅ Require stability in more frames
                stability_threshold = 8  # Must appear in 8 of last 10 frames
                current_time = time.time()
                if (
                    most_common_prediction != stable_prediction
                    and count >= stability_threshold
                    and current_time - last_prediction_time >= prediction_delay
                ):
                    stable_prediction = most_common_prediction
                    predicted_sequence.append(stable_prediction)
                    print(f"[✅] CONFIRMED: {stable_prediction} (Confidence: {confidence:.2f})")
                    capture_flash_frames = flash_duration
                    last_prediction_time = current_time
            else:
                prediction_history.clear()  # Clear if low confidence
        else:
            # 🚫 Ignore irrelevant prediction (wrong hand count)
            prediction_history.clear()
            stable_prediction = None
            capture_flash_frames = 0

        # ✅ Draw bounding box around the hand(s)
        all_landmarks = np.vstack(
            [[point.x * w, point.y * h] for hand in results.multi_hand_landmarks for point in hand.landmark]
        )
        x_min = int(np.min(all_landmarks[:, 0])) - 20
        y_min = int(np.min(all_landmarks[:, 1])) - 20
        x_max = int(np.max(all_landmarks[:, 0])) + 20
        y_max = int(np.max(all_landmarks[:, 1])) + 20

        # Keep box within frame
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, w), min(y_max, h)

        if capture_flash_frames > 0:
            box_color = (255, 255, 255)  # Flash white
            capture_flash_frames -= 1
        else:
            box_color = (0, 255, 0)  # Normal green

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 3)
        cv2.putText(image, f'{stable_prediction if stable_prediction else "..."}',
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    else:
        # 🛑 No hand detected
        prediction_history.clear()
        stable_prediction = None
        capture_flash_frames = 0
        cv2.putText(image, 'No hand detected', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 📝 Display predicted sequence below webcam feed
    sequence_text = " ".join(predicted_sequence[-10:])
    cv2.rectangle(image, (0, h - 60), (w, h), (50, 50, 50), -1)  # Background bar
    cv2.putText(image, sequence_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show webcam feed
    cv2.imshow('🖐 Hand Gesture Recognition', image)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
