import os
import csv
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# 🛡️ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📁 Paths
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, '..', 'dataset')
models_dir = os.path.join(base_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# 📊 Load dataset
data = []
labels = []

try:
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            label = os.path.splitext(filename)[0]
            file_path = os.path.join(dataset_dir, filename)

            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip header
                for row in reader:
                    values = [float(val) for val in row]
                    data.append(values)
                    labels.append(label)

    if not data:
        raise ValueError("No data found in dataset folder.")
    logging.info(f"[✅] Loaded {len(data)} samples across {len(set(labels))} gestures.")

except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit(1)

# 🧪 Prepare inputs
X = np.array(data)
y = np.array(labels)

# 🔁 Normalize landmarks
X = X / np.max(X)

# 🔤 Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 💾 Save label encoder
label_encoder_path = os.path.join(models_dir, "label_encoder.pkl")
joblib.dump(label_encoder, label_encoder_path)
logging.info(f"[💾] Label encoder saved to '{label_encoder_path}'.")

# 🔀 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
)

# 🧠 Define model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️ Train model
try:
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test))
except Exception as e:
    logging.error(f"Model training failed: {e}")
    exit(1)

# 💾 Save model
model_path = os.path.join(models_dir, "hand_gesture_cnn_model.h5")
model.save(model_path)
logging.info(f"[🎉] Model saved to '{model_path}'.")
