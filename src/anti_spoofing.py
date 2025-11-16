import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Model Definition ---
# This must be the *exact* same model structure you used for training.
# We need to define it here so PyTorch knows how to load the weights.
def get_liveness_model():
    """
    Loads a pre-trained MobileNetV2 and replaces the final
    classifier layer for our 2-class (real/spoof) task.
    """
    model = models.mobilenet_v2(pretrained=False) # No need to download pretrained weights
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) # 2 classes: real, spoof
    return model

# --- Liveness Detector Class ---
class LivenessDetector:
    def __init__(self, model_path="models/liveness_model.pth"):
        """
        Initializes the liveness detector.
        :param model_path: Path to the saved .pth model file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load the model structure
        self.model = get_liveness_model()
        
        # 2. Load the trained weights
        try:
            # Load weights onto the correct device
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Liveness model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Liveness model file not found at {model_path}")
            print("Please ensure 'liveness_model.pth' is in the 'models/' folder.")
            raise
        
        # 3. Set model to evaluation mode and move to device
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Define the image transformation
        # This MUST match the validation/test transform from your training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 5. Class names (must match training)
        self.class_names = ['spoof', 'real'] # From ImageFolder (alphabetical)

    def check_liveness(self, image):
        """
        Checks the liveness of a single image.
        :param image: A PIL Image or an OpenCV frame (which will be converted)
        :return: A string ("real" or "spoof") and the confidence score.
        """
        
        # If the image is an OpenCV frame (numpy array), convert to PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        image = image.convert("RGB")
        
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
# You can run this file directly (python src/anti_spoofing.py)
# to test if it's working.
if __name__ == "__main__":
    # Create a dummy image (e.g., all black)
    test_image = Image.new('RGB', (224, 224), color = 'black')
    
    print("Testing LivenessDetector...")
    try:
        # 1. Initialize detector
        # Make sure 'models/liveness_model.pth' exists!
        detector = LivenessDetector()
        
        # 2. Run prediction
        prediction, confidence = detector.check_liveness(test_image)
        
        print(f"\nTest prediction on dummy image: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print("\nLivenessDetector initialized and test ran successfully.")
        print("This module is ready to be imported by app.py.")
        
    except FileNotFoundError:
        print("\nTest failed: Could not find 'models/liveness_model.pth'.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")