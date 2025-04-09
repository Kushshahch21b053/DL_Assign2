import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 filters_per_layer=32, filter_size=3, filter_organization='same',
                 activation='relu', dense_neurons=128,
                 use_batchnorm=False, dropout_rate=0.0):
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
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu  
        else:
            self.activation_fn = F.relu  # Default to ReLU if invalid option

        # Calculate padding to maintain spatial dimensions after convolution
        padding = filter_size // 2

        # Calculate number of filters for each layer based on organization strategy
        if filter_organization == 'double':
            filters = [filters_per_layer * (2**i) for i in range(5)]
        elif filter_organization == 'half':
            filters = [filters_per_layer // (2**i) if filters_per_layer // (2**i) >= 8 else 8 for i in range(5)] # Minimum 8 filters
            filters.reverse() # Reverse to maintain increasing order
        else:  # 'same'
            filters = [filters_per_layer] * 5  # Same number of filters for all layers

        # --- Convolutional Blocks ---
        self.layers = nn.ModuleList()
        
        # First conv block (input → first layer)
        block1 = nn.ModuleDict({
            'conv': nn.Conv2d(input_channels, filters[0], filter_size, padding=padding)
        })
        if use_batchnorm:
            block1['bn'] = nn.BatchNorm2d(filters[0])
        block1['pool'] = nn.MaxPool2d(kernel_size=2, stride=2)
        if dropout_rate > 0:
            block1['dropout'] = nn.Dropout(dropout_rate)
        self.layers.append(block1)

        # Remaining conv blocks
        for i in range(1, 5):
            block = nn.ModuleDict({
                'conv': nn.Conv2d(filters[i-1], filters[i], filter_size, padding=padding)
            })
            if use_batchnorm:
                block['bn'] = nn.BatchNorm2d(filters[i])
            block['pool'] = nn.MaxPool2d(kernel_size=2, stride=2)
            if dropout_rate > 0:
                block['dropout'] = nn.Dropout(dropout_rate)
            self.layers.append(block)

        # Calculate the size of the feature map after all convolutions and pooling
        # Each pooling reduces dimensions by a factor of 2, so total reduction is 2^5 = 32
        height, width = input_size
        height_after_pooling = height // 32
        width_after_pooling = width // 32
        self.feature_size = height_after_pooling * width_after_pooling

        # --- Fully Connected Layers ---
        """
        First dense layer
        # Input: Flattened feature maps [batch_size, filters[-1] * feature_size]
        # Output: [batch_size, dense_neurons]
        """
        self.fc1 = nn.Linear(filters[-1] * self.feature_size, dense_neurons)
        self.fc_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Output layer
        """
        # Input: [batch_size, dense_neurons]
        # Output: [batch_size, num_classes]
        """
        self.fc2 = nn.Linear(dense_neurons, num_classes)    
    

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, height, width]   
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """

        # Process each convolutional block
        for layer_block in self.layers:
            x = layer_block['conv'](x)
            if 'bn' in layer_block:
                x = layer_block['bn'](x)
            x = self.activation_fn(x)
            x = layer_block['pool'](x)
            if 'dropout' in layer_block:
                x = layer_block['dropout'](x)

        # Flatten the tensor for the fully connected layers
        # Reshape from [batch_size, channels, height, width] to [batch_size, channels*height*width]
        x = x.view(x.size(0), -1)
        
        # --- Fully Connected Layers ---
        x = self.activation_fn(self.fc1(x))  # Apply first dense layer and activation
        x = self.fc_dropout(x)               # Apply dropout if specified
        x = self.fc2(x)                      # Apply output layer (no activation yet)
        
        return x  # Raw logits (typically use with CrossEntropyLoss which includes softmax)