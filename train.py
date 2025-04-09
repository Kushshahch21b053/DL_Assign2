import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import wandb
from tqdm import tqdm

from model import FlexibleCNN
from dataset import get_data_loaders

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', use_wandb=False):
    """
    Training function for the CNN model
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        num_epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        use_wandb: Whether to log metrics to wandb
        
    Returns:
        model: Trained model
        history: Dictionary containing training and validation metrics
    """
    model = model.to(device) # Move model to the specified device
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 

    best_val_acc = 0.0 # Initialize best validation accuracy

    for epoch in range(num_epochs):
        start_time = time.time() # Start time for the epoch

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Record statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        
        # Print metrics
        time_taken = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - {time_taken:.1f}s - '
              f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - '
              f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if use_wandb:
                torch.save(model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')

    return model, history


