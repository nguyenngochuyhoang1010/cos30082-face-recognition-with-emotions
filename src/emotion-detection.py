import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- Model Definition ---
# This class MUST be an exact copy of the one you used for training
# in the '5-emotion-detection.ipynb' notebook.
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Input: (Batch_size, 1, 48, 48)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch_size, 64, 24, 24)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch_size, 128, 12, 12)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch_size, 256, 6, 6)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(), # -> (Batch_size, 256 * 6 * 6) = 9216
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_layers(x)
        return x

# --- Emotion Detector Class ---
class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.pth"):
        """
        Initializes the emotion detector.
        :param model_path: Path to the saved .pth model file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Get class names from the FER-2013 dataset (must match training)
        # Check your notebook's Cell 3 output to be 100% sure of the order.
        # It's usually alphabetical.
        self.class_names = [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]
        num_classes = len(self.class_names)

        # 2. Load the model structure
        self.model = EmotionCNN(num_classes=num_classes)
        
        # 3. Load the trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Emotion model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Emotion model file not found at {model_path}")
            print("Please ensure 'emotion_model.pth' is in the 'models/' folder.")
            raise
        
        # 4. Set model to evaluation mode and move to device
        self.model.to(self.device)
        self.model.eval()
        
        # 5. Define the image transformation
        # This MUST match the validation/test transform from your training
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # Convert to grayscale
            transforms.Resize((48, 48)),                 # Resize to 48x48
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
        ])

    def detect_emotion(self, image):
        """
        Detects emotion from a single image.
        :param image: A PIL Image or an OpenCV frame (which will be converted)
        :return: A string (e.g., "happy") and the confidence score.
        """
        
        # If the image is an OpenCV frame (numpy array), convert to PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model outputs (logits)
            outputs = self.model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the top probability and its index
            confidence, pred_index = torch.max(probabilities, 1)
            
            # Get the predicted class name
            prediction = self.class_names[pred_index.item()]
            
        return prediction, confidence.item()

# --- Main block for testing ---
if __name__ == "__main__":
    # Create a dummy image
    test_image = Image.new('RGB', (200, 200), color = 'gray')
    
    print("Testing EmotionDetector...")
    try:
        # 1. Initialize detector
        # Make sure 'models/emotion_model.pth' exists!
        detector = EmotionDetector()
        
        # 2. Run prediction
        prediction, confidence = detector.detect_emotion(test_image)
        
        print(f"\nTest prediction on dummy image: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print("\nEmotionDetector initialized and test ran successfully.")
        print("This module is ready to be imported by app.py.")
        
    except FileNotFoundError:
        print("\nTest failed: Could not find 'models/emotion_model.pth'.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")