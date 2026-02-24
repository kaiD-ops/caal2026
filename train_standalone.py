#!/usr/bin/env python3
"""
Standalone training script for S4 Galaxy Classification Model
Based on train.ipynb
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Import project modules
from model import GalaxyClassifierS4D
from model.functions import load_data, export_model_parameters
from utils import set_pbar_style

def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device, verbose=True):
    """Train the model and validate after each epoch."""
    
    history = {"loss": [], "val_accuracy": []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, disable=not verbose, desc=f"Epoch {epoch+1}/{epochs} - Train")
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_logits = model(X_batch)
            loss = loss_fn(y_logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(y_logits, 1)
            _, true_labels = torch.max(y_batch, 1)
            train_correct += (predicted == true_labels).sum().item()
            train_total += y_batch.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, disable=not verbose, desc=f"Epoch {epoch+1}/{epochs} - Validation")
            for X_batch, y_batch in val_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_logits = model(X_batch)
                _, predicted = torch.max(y_logits, 1)
                _, true_labels = torch.max(y_batch, 1)
                val_correct += (predicted == true_labels).sum().item()
                val_total += y_batch.size(0)
        
        val_accuracy = val_correct / val_total
        
        # Log history
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return history

def main():
    """Main training function"""
    
    # Configuration
    set_pbar_style(bar_fill_color="#FFFFFF", text_color="#FFFFFF")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COLORED = False  # Use grayscale images
    RNG_SEED = 29070
    BATCH_SIZE = 64
    EPOCHS = 15
    
    print(f"Using device: {DEVICE}")
    print(f"RNG seed: {RNG_SEED}")
    print(f"Using colored images: {COLORED}")
    print()
    
    # Set seeds for reproducibility
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(RNG_SEED)
    
    # Load data
    print("Loading GalaxyMNIST dataset...")
    X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
    NUM_CLASSES = y_onehot.shape[1]
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_onehot shape: {y_onehot.shape}")
    print(f"Number of classes: {NUM_CLASSES}")
    print()
    
    # Split data
    print("Splitting data into train/validation sets...")
    x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
        X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    
    # Create dataloaders
    train_ds = TensorDataset(x_train, y_train_onehot)
    val_ds = TensorDataset(x_val, y_val_onehot)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print()
    
    # Create model
    print("Creating model...")
    model = GalaxyClassifierS4D(num_classes=NUM_CLASSES, colored=COLORED).to(DEVICE)
    print(model)
    print()
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    print(f"Starting training for {EPOCHS} epochs...")
    print()
    history = train(train_loader, val_loader, model, optimizer, loss_fn, EPOCHS, DEVICE, verbose=True)
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "galaxy_s4_model.pth")
    print("Model saved to galaxy_s4_model.pth")
    
    # Export model parameters
    print("Exporting model parameters for C/RISC-V...")
    export_model_parameters(model, "s4_model_params.h")
    print("Model parameters exported")
    
    print("\nTraining complete!")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()
