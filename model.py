import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlexibleCNN(nn.Module):
    """
    A flexible CNN architecture with 5 convolutional blocks (conv-activation-maxpool).
    It is followed by one dense layer and an output layer.
    
    The model is designed to be configurable, allowing changes to:
    - Number of filters in each layer
    - Size of filters (kernel size)
    - Activation function
    - Number of neurons in the dense layer
    
    This implementation satisfies the requirements for Part A, Question 1
    of the iNaturalist dataset classification assignment.
    """
    
    def __init__(self, input_channels=3, input_size=(224, 224), num_classes=10, 
                 filters_per_layer=32, filter_size=3,
                 activation='relu', dense_neurons=128):
        """
        Initialize the CNN model with configurable parameters.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB images)
            input_size (tuple): Input image dimensions (height, width)
            num_classes (int): Number of output classes (10 for iNaturalist subset)
            filters_per_layer (int): Number of filters in each convolutional layer
            filter_size (int): Size of the filters (kernel size)
            dense_neurons (int): Number of neurons in the dense layer
            activation (str): Activation function to use ('relu', 'gelu', 'silu', or 'mish')
        """
        super(FlexibleCNN, self).__init__()

        # Setup the activation function according to user configuration
        if activation == 'relu':
            self.activation_fn = F.relu  
        elif activation == 'gelu':
            self.activation_fn = F.gelu  
        elif activation == 'silu':
            self.activation_fn = F.silu  
        elif activation == 'mish':
            self.activation_fn = F.mish  
        else:
            self.activation_fn = F.relu  # Default to ReLU if invalid option

        # Calculate padding to maintain spatial dimensions after convolution
        padding = filter_size // 2

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,     # Input image channels (3 for RGB)
            out_channels=filters_per_layer, # Number of output feature maps
            kernel_size=filter_size,        # Kernel size (e.g., 3x3)
            padding=padding                 # Padding to maintain spatial dimensions
                         )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(
            in_channels=filters_per_layer, # Input channels from previous layer
            out_channels=filters_per_layer, # Number of output feature maps
            kernel_size=filter_size,
            padding=padding
                         )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(
            in_channels=filters_per_layer, # Input channels from previous layer
            out_channels=filters_per_layer, # Number of output feature maps
            kernel_size=filter_size,
            padding=padding
                         )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 4 ---
        self.conv4 = nn.Conv2d(
            in_channels=filters_per_layer, # Input channels from previous layer
            out_channels=filters_per_layer, # Number of output feature maps
            kernel_size=filter_size,
            padding=padding
                         )

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 5 ---
        self.conv5 = nn.Conv2d(
            in_channels=filters_per_layer, # Input channels from previous layer
            out_channels=filters_per_layer, # Number of output feature maps
            kernel_size=filter_size,
            padding=padding
                         )

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the feature map after all convolutions and pooling
        # Each pooling reduces dimensions by a factor of 2, so total reduction is 2^5 = 32
        height, width = input_size
        height_after_pooling = height // 32
        width_after_pooling = width // 32
        self.feature_size = height_after_pooling * width_after_pooling

        # --- Fully Connected Layers ---
        """
        First dense layer
        # Input: Flattened feature maps [batch_size, filters_per_layer * feature_size]
        # Output: [batch_size, dense_neurons]
        """
        self.fc1 = nn.Linear(
            in_features=filters_per_layer * self.feature_size, 
            out_features=dense_neurons
        )

        # Output layer
        """
        # Input: [batch_size, dense_neurons]
        # Output: [batch_size, num_classes]
        """
        self.fc2 = nn.Linear(
            in_features=dense_neurons, 
            out_features=num_classes
        )
    

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, height, width]   
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """

        # --- Block 1: Convolution + Activation + MaxPool ---
        x = self.activation_fn(self.conv1(x))  # Apply convolution and activation
        x = self.pool1(x)                      # Apply max pooling
        
        # --- Block 2: Convolution + Activation + MaxPool ---
        x = self.activation_fn(self.conv2(x))
        x = self.pool2(x)
        
        # --- Block 3: Convolution + Activation + MaxPool ---
        x = self.activation_fn(self.conv3(x))
        x = self.pool3(x)
        
        # --- Block 4: Convolution + Activation + MaxPool ---
        x = self.activation_fn(self.conv4(x))
        x = self.pool4(x)
        
        # --- Block 5: Convolution + Activation + MaxPool ---
        x = self.activation_fn(self.conv5(x))
        x = self.pool5(x)

        # Flatten the tensor for the fully connected layers
        # Reshape from [batch_size, channels, height, width] to [batch_size, channels*height*width]
        x = x.view(x.size(0), -1)
        
        # --- Fully Connected Layers ---
        x = self.activation_fn(self.fc1(x))  # Apply first dense layer and activation
        x = self.fc2(x)                      # Apply output layer (no activation yet)
        
        return x  # Raw logits (typically use with CrossEntropyLoss which includes softmax)