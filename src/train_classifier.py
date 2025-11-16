import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import create_classification_dataloaders
from models import get_face_classifier

# --- Configuration ---
TRAIN_DIR = 'data/classification_data/train_data'
VAL_DIR = 'data/classification_data/val_data'
MODEL_SAVE_PATH = 'models/face_classifier.pth'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print('Training complete!')
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, num_classes = create_classification_dataloaders(
        TRAIN_DIR, VAL_DIR, BATCH_SIZE
    )
    print(f"Found {num_classes} classes.")

    model = get_face_classifier(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model = train_model(model, dataloaders, criterion, optimizer, device, NUM_EPOCHS)
    
    # Save the final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()