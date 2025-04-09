import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import FlexibleCNN
from dataset import get_data_loaders
from train import train_model, evaluate_model
from config import get_config

def main():
    # Get configuration
    args = get_config()

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(
        data_dir='nature_12K\inaturalist_12K',
        batch_size=args.batch_size,
        val_split=args.val_split,
        apply_augmentation=args.augmentation,
        random_seed=args.random_seed
    )

    # Create model
    model = FlexibleCNN(
        input_channels=args.input_channels,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes,
        filters_per_layer=args.filters_per_layer,
        filter_size=args.filter_size,
        filter_organization=args.filter_organization,
        activation=args.activation,
        dense_neurons=args.dense_neurons,
        use_batchnorm=args.use_batchnorm,
        dropout_rate=args.dropout_rate
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    # Train the model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        use_wandb=args.use_wandb
    )

if __name__ == "__main__":
    main()
