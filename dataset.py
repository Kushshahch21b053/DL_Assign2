import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAME_TO_IDX = {
    "Amphibia": 0,
    "Animalia": 1,
    "Arachnida": 2,
    "Aves": 3,
    "Fungi": 4,
    "Insecta": 5,
    "Mammalia": 6,
    "Mollusca": 7,
    "Plantae": 8,
    "Reptilia": 9
}

class iNaturalistDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, val_split=0.2):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load the training or test set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.val_split = val_split

        # Set the approporiate directory
        if self.train:
            self.data_dir = os.path.join(root_dir, 'train')
        else:
            self.data_dir = os.path.join(root_dir, 'val')

        # Get and store all image paths and labels
        self.image_paths = []
        self.labels = []

        # List all class folders
        for class_name, class_idx in CLASS_NAME_TO_IDX.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_data_loaders(data_dir, batch_size=32, val_split=0.2, apply_augmentation=False, random_seed=42):
    """
        Create training and validation data loaders

    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for training and validation
        val_split (float): Proportion of training data to use for validation
        apply_augmentation (bool): Whether to apply data augmentation
        random_seed (int): Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    
    # Define basic transformations for training and validation
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #  Define data augmentation transformations, if augmentaion is enabled
    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Random rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Color jitter 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = basic_transform
    
    # Load the full training dataset
    full_train_dataset = iNaturalistDataset(root_dir=data_dir, transform=train_transform, train=True)

    # Create train/validation split to ensure class balance
    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)),
        test_size=val_split,
        stratify=full_train_dataset.labels,
        random_state=random_seed
    )

    # Create subsets for training and validation
    train_dataset = Subset(full_train_dataset, train_indices)

    # For validation, we can use the basic transformation as training, but without augmentation
    val_dataset = iNaturalistDataset(root_dir=data_dir, transform=basic_transform, train=True)
    val_dataset = Subset(val_dataset, val_indices)

    # Load the test dataset
    test_dataset = iNaturalistDataset(root_dir=data_dir, transform=basic_transform, train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # Shuffle for training
    # No shuffling for validation and test loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
    