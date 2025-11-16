import torch
import torch.nn as nn
from torchvision import models

# --- Model 1: Face Classifier (ResNet-18) ---
def get_face_classifier(num_classes):
    """
    Loads a pre-trained ResNet-18 and replaces the final
    fully connected layer for classification.
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Model 2: Metric Learner (ResNet-18 base) ---
class EmbeddingNet(nn.Module):
    """
    The EmbeddingNet for metric learning.
    Loads a ResNet-18 base and replaces the final
    layer with a new embedding layer.
    """
    def __init__(self, embedding_dim=512):
        super(EmbeddingNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, embedding_dim)
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.model(x)
        x = self.l2_norm(x, p=2, dim=1) # Normalize embeddings
        return x

# --- Model 3: Liveness Detector (MobileNetV2) ---
def get_liveness_model():
    """
    Loads a pre-trained MobileNetV2 and replaces the final
    classifier layer for 2-class (real/spoof) task.
    """
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) # 2 classes
    return model

# --- Model 4: Emotion Detector (Custom CNN) ---
class EmotionCNN(nn.Module):
    """
    The custom CNN for 48x48 grayscale emotion detection.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_layers(x)
        return x