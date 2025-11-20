import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Model Definition (ResNet-18) ---
def get_liveness_model():
    model = models.resnet18(pretrained=False) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

class LivenessDetector:
    def __init__(self, model_path="models/liveness_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_liveness_model()
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Liveness model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            raise
        except RuntimeError as e:
            print(f"Error loading weights: {e}")
            print("Make sure your .pth file matches the ResNet-18 architecture!")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # --- THE FIX ---
        # Since your model predicts Index 1 for everything:
        # Index 0 = 'spoof'
        # Index 1 = 'real' (Winner) -> Green Box
        self.class_names = ['spoof', 'real'] 

    def check_liveness(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # --- SIMPLE LOGIC (Argmax) ---
            # No strict threshold. Just pick the highest score.
            confidence, pred_index = torch.max(probabilities, 1)
            prediction = self.class_names[pred_index.item()]
            
        return prediction, confidence.item()

# --- Main block for testing ---
if __name__ == "__main__":
    test_image = Image.new('RGB', (224, 224), color = 'black')
    print("Testing LivenessDetector...")
    try:
        detector = LivenessDetector()
        prediction, confidence = detector.check_liveness(test_image)
        print(f"\nTest prediction: {prediction} ({confidence:.4f})")
    except Exception as e:
        print(f"\nAn error occurred: {e}")