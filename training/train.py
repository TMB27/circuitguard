# Simple training script for PCB defect classifier
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from efficientnet_model import create_model
from dataset import create_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(data_folder, num_epochs=50, batch_size=32, learning_rate=0.0001):
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_folder, batch_size=batch_size
    )
    
    # Load class mapping
    with open(os.path.join(data_folder, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=num_classes)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create checkpoint folder
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    # Test final model
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model


# Run training
if __name__ == "__main__":
    data_folder = "../data/splits"
    
    train_model(
        data_folder=data_folder,
        num_epochs=10,
        batch_size=32,
        learning_rate=0.0001
    )

