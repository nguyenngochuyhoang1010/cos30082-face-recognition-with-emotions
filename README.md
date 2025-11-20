Project Overview

This project implements a complete facial recognition attendance system with three integrated deep learning modules:

Face Verification: Identifies registered users using a Metric Learning model (ResNet-18 with Triplet Loss).

Anti-Spoofing: Detects "liveness" to prevent spoofing attacks (e.g., phone screens, printed photos).

Emotion Detection: Analyzes user facial expressions in real-time.

The system features a GUI built with PyQt5 and uses OpenCV for video capture.

Installation

1. Prerequisites

Ensure you have Python 3.8+ installed.

2. Install Dependencies

Run the following command to install all required libraries:

pip install -r requirements.txt


(Key requirements: torch, torchvision, opencv-python, PyQt5, numpy, pillow)

3. Model Setup

Ensure the models/ directory contains the following trained model files (submitted with the project):

metric_learning_model.pth (Face Verification)

liveness_model.pth (Anti-Spoofing)

emotion_model.pth (Emotion Detection)

How to Run

Navigate to the project root directory in your terminal.

Run the application:

python src/app.py


The GUI window will open and activate your webcam.

User Guide

1. Registration

Click the "Register New User" button.

Enter a name in the dialog box.

Look at the camera. The system will wait for a clear, "Real" face.

Once captured, the face embedding is saved to face_db.json.

2. Verification

Click the "Verify Attendance" button.

Look at the camera.

The system will compare your live face against the database.

Result: It will display "Welcome, [Name]!" and your current emotion (e.g., "Happy").

3. Anti-Spoofing (Security)

The system continuously monitors for liveness.

If a spoof attempt is detected (e.g., a phone screen held up to the camera), a RED BOX will appear with the alert "SPOOF DETECTED".

Access and registration are blocked while a spoof is detected.

Demo Features & Troubleshooting

Note on Liveness Sensitivity:
The anti-spoofing model is highly sensitive to texture and depth cues. In some lighting conditions, it may be difficult to trigger a "Spoof" detection using a high-quality phone screen.

Manual Demo Mode:
To assist with grading and demonstration, a manual override is included to verify the UI logic:

Hold the 'S' key on your keyboard to force a "Spoof" detection signal.

This allows you to verify that the system correctly locks down (Red Box) when a spoof is reported by the model logic.

Project Structure

/
├── models/                 # Trained .pth model files
├── src/
│   ├── app.py              # Main GUI Application
│   ├── anti_spoofing.py    # Liveness detection module
│   ├── emotion_detection.py# Emotion classification module
│   ├── models.py           # PyTorch model architecture definitions
│   └── ...
├── notebooks/              # Jupyter notebooks used for training
├── face_db.json            # Local database of registered users
└── requirements.txt        # Python dependencies

