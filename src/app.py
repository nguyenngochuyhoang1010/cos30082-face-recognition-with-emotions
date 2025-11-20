import sys
import os
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, QSize

# --- Project Module Imports ---
from anti_spoofing import LivenessDetector
from emotion_detection import EmotionDetector
from models import EmbeddingNet 

# --- Constants ---
DB_PATH = 'face_db.json'
VERIFICATION_THRESHOLD = 0.8  # Confidence threshold for a match (0.0 to 1.0)
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# --- Face Verifier Class ---
class FaceVerifier:
    def __init__(self, model_path="models/metric_learning_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load the model structure (EmbeddingNet from models.py)
        self.model = EmbeddingNet()
        
        # 2. Load the trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Face verification model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Face verifier model not found at {model_path}")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Define the image transformation (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image):
        """
        Gets the embedding vector for a single PIL image.
        :param image: A PIL Image
        :return: A torch.Tensor (1D embedding vector)
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu().squeeze()


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Recognition Attendance System")
        self.setGeometry(100, 100, 800, 700)
        
        # --- App State ---
        self.mode = "idle"  # "idle", "register", "verify"
        self.register_name = ""
        self.face_db = self.load_database()
        
        # --- Demo Mode Flag ---
        self.force_spoof = False

        # --- Load All Models ---
        print("Loading models...")
        try:
            self.verifier = FaceVerifier()
            self.liveness_detector = LivenessDetector()
            self.emotion_detector = EmotionDetector()
            self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if self.face_cascade.empty():
                raise IOError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")
            print("All models loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load models. {e}")
            sys.exit()

        # --- GUI Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Video feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(QSize(640, 480))
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #555; background-color: black;")
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        
        # Status label
        self.status_label = QLabel("Status: Ready", self)
        self.status_label.setFont(QFont("Inter", 16))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        
        # Button layout
        self.button_layout = QHBoxLayout()
        self.register_button = QPushButton("Register New User", self)
        self.register_button.setFont(QFont("Inter", 14))
        self.register_button.clicked.connect(self.start_registration)
        
        self.verify_button = QPushButton("Verify Attendance", self)
        self.verify_button.setFont(QFont("Inter", 14))
        self.verify_button.clicked.connect(self.start_verification)
        
        self.button_layout.addWidget(self.register_button)
        self.button_layout.addWidget(self.verify_button)
        self.layout.addLayout(self.button_layout)
        
        # --- Initialize Webcam ---
        self.init_webcam()

    def init_webcam(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.status_label.setText("Error: Could not open webcam.")
            return
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update ~33 times per second

    def load_database(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'r') as f:
                return json.load(f)
        return []

    def save_database(self):
        with open(DB_PATH, 'w') as f:
            json.dump(self.face_db, f, indent=4)

    def start_registration(self):
        name, ok = QInputDialog.getText(self, "Register User", "Enter your name:")
        if ok and name:
            self.register_name = name
            self.mode = "register"
            self.status_label.setText(f"Registering {name}. Look at the camera...")
        else:
            self.status_label.setText("Registration cancelled.")

    def start_verification(self):
        self.mode = "verify"
        self.status_label.setText("Verifying... Look at the camera...")

    # --- KEYBOARD EVENTS FOR DEMO MODE ---
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.force_spoof = True
            print("Demo Mode: SPOOF FORCED")

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_S:
            self.force_spoof = False
            print("Demo Mode: Normal Operation")
    # -------------------------------------

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Create a copy for processing
        processing_frame = frame.copy()
        
        # Convert for face detection
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # We'll use this as the final display frame
        display_frame = processing_frame

        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            
            # --- CRITICAL: Convert face crop to PIL Image (RGB) ---
            face_crop_cv = processing_frame[y:y+h, x:x+w]
            
            # Avoid crash if face crop is empty
            if face_crop_cv.size == 0:
                return
                
            face_crop_rgb = cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2RGB)
            face_crop_pil = Image.fromarray(face_crop_rgb)
            
            # --- 1. Liveness Check ---
            liveness, liveness_conf = self.liveness_detector.check_liveness(face_crop_pil)
            
            # --- DEMO OVERRIDE ---
            if self.force_spoof:
                liveness = "spoof"
            # ---------------------
            
            if liveness == "spoof":
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(display_frame, "SPOOF DETECTED", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.status_label.setText("Status: Spoof Detected! Access Denied.")
                
                # Reset mode for security
                if self.mode != "idle":
                    self.mode = "idle"
            
            elif liveness == "real":
                # --- 2. Emotion Detection (always run if real) ---
                emotion, emotion_conf = self.emotion_detector.detect_emotion(face_crop_pil)
                
                # Draw green rectangle for "real"
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Emotion: {emotion}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- 3. Handle App Mode (Register or Verify) ---
                if self.mode == "register":
                    self.process_registration(face_crop_pil)
                
                elif self.mode == "verify":
                    self.process_verification(face_crop_pil, emotion)
        
        else:
            if self.mode == "register" or self.mode == "verify":
                self.status_label.setText("No face detected...")

        # Display the final frame
        self.display_image(display_frame)

    def process_registration(self, face_pil):
        # Get embedding for the face
        embedding = self.verifier.get_embedding(face_pil)
        
        # Add to database
        self.face_db.append({
            "name": self.register_name,
            "embedding": embedding.cpu().numpy().tolist() # Store as list
        })
        self.save_database()
        
        self.status_label.setText(f"Successfully registered {self.register_name}!")
        print(f"Registered {self.register_name} with embedding.")
        self.mode = "idle"

    def process_verification(self, face_pil, emotion):
        if not self.face_db:
            self.status_label.setText("No users registered. Please register first.")
            self.mode = "idle"
            return
            
        # Get embedding of the current face
        current_embedding = self.verifier.get_embedding(face_pil)
        
        # Compare to all embeddings in the database
        best_score = -1.0
        best_name = "Unknown"
        
        for user in self.face_db:
            db_embedding = torch.tensor(user["embedding"])
            
            # Calculate Cosine Similarity
            score = F.cosine_similarity(current_embedding.unsqueeze(0), 
                                        db_embedding.unsqueeze(0))
            
            if score.item() > best_score:
                best_score = score.item()
                best_name = user["name"]
        
        # Check against threshold
        if best_score >= VERIFICATION_THRESHOLD:
            self.status_label.setText(f"Welcome, {best_name}! (Emotion: {emotion})")
        else:
            self.status_label.setText(f"User Not Recognized. (Emotion: {emotion})")
        
        print(f"Verification: Best match {best_name} with score {best_score:.4f}")
        self.mode = "idle"

    def display_image(self, img):
        # Convert OpenCV BGR image to QPixmap for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        # Clean up
        self.timer.stop()
        self.capture.release()
        print("Application closing.")
        event.accept()


# --- Run the Application ---
if __name__ == "__main__":
    # Ensure the models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created 'models' directory. Please add your .pth files.")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())