#!/usr/bin/env python3
import argparse
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def get_input_args():
    parser = argparse.ArgumentParser(
        description='Train a new network on a dataset of images using ResNet18.'
    )
    parser.add_argument('data_dir', type=str,
                        help='Directory of the dataset (must contain "train", "valid", and "test" subfolders)')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save the checkpoint')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    return parser.parse_args()

def load_data(data_dir):
    # Define directories for train, validation, and test sets
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')
    
    # Define transforms for training with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation and testing (no augmentation)
    test_valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    
    # Create DataLoaders with increased batch size, num_workers, and pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Increased batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader, train_dataset

def build_model(num_classes=102):
    """
    Load a pretrained ResNet18 model and modify its fully connected layer.
    Assumes that the dataset has num_classes classes.
    """
    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Get the number of input features for the final fc layer
    num_ftrs = model.fc.in_features
    
    # Replace the fc layer with a new classifier
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        # Using tqdm for a progress bar over training batches
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            progress_bar.set_postfix({'Train Loss': f'{running_loss/((progress_bar.n or 1)):.3f}'})
        
        # After each epoch, evaluate on the validation set
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        epoch_valid_loss = valid_loss / len(valid_loader)
        epoch_accuracy = correct.double() / total
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {running_loss/len(train_loader):.3f}, "
              f"Valid Loss: {epoch_valid_loss:.3f}, "
              f"Accuracy: {epoch_accuracy:.3f}")
        
        model.train()  # Set back to training mode
    return model

def save_checkpoint(model, train_dataset, epochs, save_dir):
    checkpoint = {
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'checkpoint_resnet18.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    args = get_input_args()
    
    # Set device to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the data
    train_loader, valid_loader, test_loader, train_dataset = load_data(args.data_dir)
    
    # Build the model (assume 102 classes; adjust if needed)
    model = build_model(num_classes=102)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train the model
    start_time = time.time()
    model = train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=args.epochs)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time//60:.0f} minutes {elapsed_time%60:.0f} seconds")
    
    # Save the checkpoint
    save_checkpoint(model, train_dataset, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()
