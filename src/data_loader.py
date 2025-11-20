import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# --- Transformations ---
IMAGE_SIZE_CLASSIFIER = (224, 224)
IMAGE_SIZE_EMOTION = (48, 48)

# Standard transform for ResNet/MobileNet (3-channel)
# Updated 'train' transform with stronger augmentation for robustness
transform_3_channel = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE_CLASSIFIER),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Vary lighting
        transforms.RandomGrayscale(p=0.1), # Force texture learning
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE_CLASSIFIER),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Transform for Emotion model (1-channel)
transform_1_channel = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE_EMOTION),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE_EMOTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
}


def create_classification_dataloaders(train_dir, val_dir, batch_size):
    """Creates dataloaders for a simple classification task."""
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform_3_channel['train']),
        'val': datasets.ImageFolder(val_dir, transform_3_channel['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    num_classes = len(image_datasets['train'].classes)
    return dataloaders, num_classes


# --- TripletFaceDataset Class (from our notebook) ---
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_paths = {}
        self.class_list = []

        print("Indexing dataset...")
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            self.class_list.append(class_name)
            self.class_to_paths[class_name] = []
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.class_to_paths[class_name].append(img_path)
        
        full_anchor_list = []
        for class_name, paths in self.class_to_paths.items():
            for path in paths:
                full_anchor_list.append((path, class_name))
                
        print(f"Found {len(self.class_list)} classes, {len(full_anchor_list)} total images.")

        if subset_fraction < 1.0:
            num_to_sample = int(len(full_anchor_list) * subset_fraction)
            indices = np.random.choice(len(full_anchor_list), num_to_sample, replace=False)
            self.anchor_list = [full_anchor_list[i] for i in indices]
            print(f"Using a subset of {num_to_sample} images.")
        else:
            self.anchor_list = full_anchor_list

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, index):
        anchor_path, anchor_class = self.anchor_list[index]
        
        positive_list = self.class_to_paths[anchor_class]
        positive_path = anchor_path
        while positive_path == anchor_path and len(positive_list) > 1:
            positive_path = random.choice(positive_list)
        
        negative_class = random.choice(self.class_list)
        while negative_class == anchor_class:
            negative_class = random.choice(self.class_list)
            
        negative_path = random.choice(self.class_to_paths[negative_class])
        
        anchor_img = self.load_image(anchor_path)
        positive_img = self.load_image(positive_path)
        negative_img = self.load_image(negative_path)
        
        return anchor_img, positive_img, negative_img

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img