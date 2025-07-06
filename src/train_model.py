import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# 📁 Dataset directory (where your gesture CSVs are stored)
dataset_dir = "dataset"

# 🏷 Prepare data and labels
data = []
labels = []

# 🔄 Load all CSV files in dataset folder
for filename in os.listdir(dataset_dir):
    if filename.endswith(".csv"):
        label = os.path.splitext(filename)[0]  # e.g., A.csv -> label = 'A'
        file_path = os.path.join(dataset_dir, filename)

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                landmark_coords = [float(val) for val in row]  # Convert all values to float
                data.append(landmark_coords)
                labels.append(label)

print(f"[✅] Loaded {len(data)} samples across {len(set(labels))} gestures.")

# 📊 Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# 📏 Normalize landmark data (scale between 0 and 1)
X = X / np.max(X)

# 🔤 Encode labels (A-Z, 0-9, space)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 💾 Save label encoder for later use in prediction
joblib.dump(label_encoder, "label_encoder.pkl")
print("[💾] Label encoder saved as 'label_encoder.pkl'.")

# 📈 Split into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# 🧠 Build CNN model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])

# ⚙ Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋 Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test))

# 💾 Save trained model
model.save("hand_gesture_cnn_model.h5")
print("[🎉] CNN model saved as 'hand_gesture_cnn_model.h5'.")