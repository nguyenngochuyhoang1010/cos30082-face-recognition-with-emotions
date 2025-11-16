import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import TripletFaceDataset, transform_3_channel
from models import EmbeddingNet

# --- Configuration ---
TRAIN_DIR = 'data/classification_data/train_data'
MODEL_SAVE_PATH = 'models/metric_learning_model.pth'
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 15
MARGIN = 1.0
SUBSET_FRACTION = 0.2 # Use 20% of data for faster training

def train_triplet_model(model, loader, criterion, optimizer, device, num_epochs):
    print("Starting metric learning training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        batch_iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for anchor, positive, negative in batch_iterator:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_iterator.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training complete!')
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TripletFaceDataset(
        root_dir=TRAIN_DIR,
        transform=transform_3_channel['train'],
        subset_fraction=SUBSET_FRACTION
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    model = EmbeddingNet().to(device)
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model = train_triplet_model(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()