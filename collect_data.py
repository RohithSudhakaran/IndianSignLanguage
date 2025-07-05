import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# 🖐 Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 📁 Directory to save datasets
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

# 🎥 Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌] Could not access webcam.")
    exit()

while True:
    # 📝 Ask for gesture label
    gesture_label = input("\n🤚 Enter gesture label (A-Z, 0-9, space) or type 'exit' to quit: ").strip()
    if gesture_label.lower() == "exit":
        print("[🚪] Exiting gesture collection.")
        break
    if gesture_label.lower() == "space":
        gesture_label = "space"

    # 📄 Path for this gesture's CSV file
    csv_file = os.path.join(dataset_dir, f"{gesture_label}.csv")

    # ⚠️ Check if file exists and ask to replace
    if os.path.exists(csv_file):
        choice = input(f"⚠️ Data for '{gesture_label}' already exists. Replace it? (y/n): ").strip().lower()
        if choice == 'y':
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
                writer.writerow(header)
            print(f"[📝] Existing data for '{gesture_label}' replaced.")
        else:
            print(f"[➕] New samples will be added to existing data for '{gesture_label}'.")
    else:
        # Create new file with header
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            writer.writerow(header)

    try:
        max_samples = int(input("🔢 Enter number of samples to collect (excluding first 50): "))
    except ValueError:
        print("[⚠️] Invalid input. Defaulting to 100 samples.")
        max_samples = 100

    print(f"\n[ℹ️] Get ready to show gesture '{gesture_label}'. Collecting {max_samples} samples (skipping first 50).")
    time.sleep(1)

    # ⏳ Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    raw_sample_count = 0  # total frames processed
    saved_sample_count = 0  # actual samples saved

    while saved_sample_count < max_samples:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # ⚠️ DO NOT FLIP THE IMAGE (keep original camera feed for accuracy)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 📥 Extract landmark coordinates
                h, w, _ = image.shape
                landmarks = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark])
                flattened_landmarks = landmarks.flatten().tolist()

                raw_sample_count += 1  # increment raw frame count

                if raw_sample_count > 50:  # Skip first 50 frames
                    # 📝 Save to CSV
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(flattened_landmarks)

                    saved_sample_count += 1
                    print(f"[✅] Saved sample {saved_sample_count}/{max_samples} for '{gesture_label}'")
                else:
                    print(f"[⏳] Warming up... Skipping sample {raw_sample_count}/50")

        # 🖥 Show progress on webcam
        cv2.putText(image, f'Gesture: {gesture_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(image, f'Samples: {saved_sample_count}/{max_samples}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        cv2.imshow('🖐 Hand Gesture Collection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[⚠️] Manual quit detected for this gesture.")
            break

    print(f"\n[🎉] Collected {saved_sample_count} samples for gesture '{gesture_label}'.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[✅] Dataset collection finished.")
