import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from model import FlexibleCNN
from dataset import get_data_loaders, CLASS_NAME_TO_IDX
from train import train_model
from config import get_config

if __name__ == '__main__':
    """
    Best Hyperparameters
    """
    # Best parameters from wandb
    filters_per_layer = 32          
    filter_organization = 'same'  
    activation = 'relu'            
    learning_rate = 0.001
    use_batchnorm = True            
    dropout_rate = 0
    augmentation = False              

    # Training hyperparameters
    num_classes = 10
    input_channels = 3
    input_size = (224, 224)
    batch_size = 32
    dense_neurons = 128             
    epochs = 10
    filter_size = 3                     
    weight_decay = 0.0
    random_seed = 42
    optimizer_choice = 'adam'       
    use_wandb = False               


    """
    Data Loading
    """
    args = get_config()
    data_dir = args.data_dir

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        apply_augmentation=augmentation,
        random_seed=random_seed
    )
    test_dataset = test_loader.dataset

    """
    Initialize and train best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FlexibleCNN(
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        filters_per_layer=filters_per_layer,
        filter_size=filter_size,
        filter_organization=filter_organization,
        activation=activation,
        dense_neurons=dense_neurons,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate
    )
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer according to our hyperparameter choice
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # Start training the model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        use_wandb=use_wandb
    )

    """
    Evaluate on test set
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100* correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    """
    Create a Grid of Sample Test Images with predictions
    """
    # Create a dictionary to store indices for each class (0 to 9).
    # We will collect 3 indices per class.
    """
    Create a Balanced Prediction Grid of Sample Test Images with Predictions:
    3 images per class (total of 30 images)
    """

    # Build a reverse mapping: class index -> class name.
    idx_to_class = {v: k for k, v in CLASS_NAME_TO_IDX.items()}

    # Create a dictionary to collect indices for each class (0 to 9).
    selected_indices = {c: [] for c in range(10)}

    # Iterate over the entire test_dataset to collect 3 indices per class.
    for idx in range(len(test_dataset)):
        # Get the label for the image.
        _, label = test_dataset[idx]
        # Append the index if the class hasn't reached 3 images yet.
        if len(selected_indices[label]) < 3:
            selected_indices[label].append(idx)
        # Once we have 3 images for every class, break out.
        if all(len(indices) == 3 for indices in selected_indices.values()):
            break

    print("Selected indices per class:", selected_indices)

    # Order indices from class 0 to class 9.
    ordered_indices = []
    for c in range(10):
        ordered_indices.extend(selected_indices[c])

    # Retrieve the images and labels for these indices.
    selected_images = []
    selected_labels = []
    for idx in ordered_indices:
        img, lbl = test_dataset[idx]
        selected_images.append(img)
        selected_labels.append(lbl)

    # Stack the images into a single tensor.
    selected_images = torch.stack(selected_images)
    selected_images = selected_images.to(device)

    # Run the model predictions on the selected images.
    model.eval()
    with torch.no_grad():
        outputs = model(selected_images)
        _, preds = torch.max(outputs, 1)

    # Bring the images and predictions back to CPU for visualization.
    selected_images = selected_images.cpu()
    preds = preds.cpu()

    # Set grid dimensions: 10 rows (each for one class) and 3 columns (3 images per class).
    rows, cols = 10, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # Denormalize the images.
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # Loop through the 30 selected images and plot them.
    for i in range(rows * cols):
        img = inv_normalize(selected_images[i])
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        true_label = idx_to_class[selected_labels[i]]
        pred_label = idx_to_class[preds[i].item()]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig('test_predictions_grid.png')
    plt.show()
