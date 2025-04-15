import argparse
import wandb

def get_config():
    """
    Parse command line arguments and initialize wandb configuration.
    
    Returns:
        args: Parsed command line arguments with model configuration
    """
    parser = argparse.ArgumentParser(description='CNN for iNaturalist Classification')
    
    # Model configuration
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels (3 for RGB images)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--filters_per_layer', type=int, default=32,
                        help='Number of filters in each convolutional layer')
    parser.add_argument('--filter_size', type=int, default=3,
                        help='Size of the convolutional filters')
    parser.add_argument('--filter_organization', type=str, default='same',
                        choices=['same', 'double', 'half'],
                        help='Strategy for distributing filters across layers')
    parser.add_argument('--dense_neurons', type=int, default=128,
                        help='Number of neurons in the dense layer')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'silu', 'mish','leaky_relu'],
                        help='Activation function to use')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                        help='Input image dimensions (height, width)')
    parser.add_argument('--use_batchnorm', action='store_true',
                        help='Whether to use batch normalization')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate (0.0 means no dropout)')
    parser.add_argument('--augmentation', action='store_true',
                        help='Whether to use data augmentation')
    
    # Training configuration - these will be needed for later parts of the assignment
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for regularization')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'],
                    help='Optimizer to use (adam or sgd)')

    # Wandb configuration
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='DL_Assign2',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity name')
    
    args = parser.parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
        # Update args with wandb config for hyperparameter sweeps
        args = argparse.Namespace(**wandb.config)
    
    return args
