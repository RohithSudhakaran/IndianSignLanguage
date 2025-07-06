# 🖐 Indian Sign Language Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Status: Stable](https://img.shields.io/badge/Status-Stable-brightgreen)

**Indian Sign Language Detector** is a real-time hand gesture recognition system that translates Indian Sign Language (ISL) into text using a webcam. It combines **MediaPipe** for hand tracking and a **TensorFlow CNN** for gesture classification.

*This project is lightweight, modular, and beginner-friendly. You can easily collect your own dataset, retrain the model, and extend it for more gestures.*

---

## ✨ Features

- 📷 **Real-Time Detection**: Recognizes ISL gestures instantly using your webcam.  
- 🖐 **One-Hand & Two-Hand Gestures**: Supports complex signs requiring multiple hands.  
- 🧠 **Custom CNN Model**: Trained on user-collected datasets for high accuracy.  
- 🗃 **Dataset Collector**: Easily create and expand your gesture dataset.  
- 🔄 **Smooth Predictions**: Includes stability checks to reduce flickering.  
- 🛠 **Extensible Design**: Add new gestures and retrain without rewriting code.  

---

## 📂 Folder Structure

```
indian-sign-language/
│
├── src/                  # Source code
│   ├── collect_data.py   # Script to collect gesture dataset
│   ├── train_model.py    # Script to train CNN model
│   └── main.py           # Real-time gesture detection
│
├── models/               # Saved model and label encoder
│   ├── hand_gesture_cnn_model.h5
│   └── label_encoder.pkl
│
├── dataset/              # Collected gesture CSV files
│
├── requirements.txt      # Project dependencies
├── .gitignore            # Files to ignore in Git
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 🖥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/indian-sign-language.git
   cd indian-sign-language
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Activate on Linux/Mac
   source venv/bin/activate
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

### 📝 Usage

#### 1️⃣ Collect Dataset
Run this script to capture gesture data:  
```bash
python src/collect_data.py
```

#### 2️⃣ Train the Model
After collecting data, train your CNN model:  
```bash
python src/train_model.py
```

#### 3️⃣ Run Gesture Detection
Start real-time gesture recognition:  
```bash
python src/main.py
```

---

## 🛠 Built With

- [MediaPipe](https://google.github.io/mediapipe/) – Hand tracking
- [TensorFlow](https://www.tensorflow.org/) – CNN model
- [OpenCV](https://opencv.org/) – Webcam access
- [NumPy](https://numpy.org/) – Data processing

---

## 🙌 Contributing

Contributions are welcome!  
1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/awesome-feature`)  
3. Commit your changes (`git commit -m 'Add some awesome feature'`)  
4. Push to the branch (`git push origin feature/awesome-feature`)  
5. Open a Pull Request  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

- **Rohith S**  
  [GitHub](https://github.com/<your-username>)
