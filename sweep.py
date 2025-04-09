import wandb
import os
import subprocess
import argparse

def create_sweep_configuration():
    """
    Create configuration for wandb hyperparameter sweep
    """

    sweep_config = {
        'method': 'bayes', # Bayesian optimization
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'filters_per_layer': {
                'values': [32, 64]
            },
            'filter_organization': {
                'values': ['same', 'double', 'half']
            },
            'activation': {
                'values': ['leaky_relu', 'gelu']
            },
            'dropout_rate': {
                'values': [0.0, 0.2, 0.3]
            },
            'learning_rate': {
                'values': [0.001, 0.01]
            },
            'use_batchnorm': {
                'values': [True, False]
            },
            'augmentation': {
                'values': [True, False]
            }
        }
    }

    return sweep_config

def run_sweep_agent():
    """
    Function to run the sweep agent
    """
    def train():
        # Initialize a new wandb run
        wandb.init()

        # Construct command with all wandb config parameters
        command = ['python', 'main.py', '--use_wandb']
        
        # Add all config parameters from wandb
        for key, value in wandb.config.items():
            if isinstance(value, bool):
                if value:
                    command.append(f'--{key}')
            else:
                command.append(f'--{key}')
                command.append(str(value))
        
        # Run the training script as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Print output for debugging
        print(stdout.decode())
        if stderr:
            print("Errors:")
            print(stderr.decode())
    
    return train

def main():
    parser = argparse.ArgumentParser(description='Run wandb sweep for iNaturalist CNN')
    parser.add_argument('--create', action='store_true', help='Create a new sweep')
    parser.add_argument('--sweep_id', type=str, default=None, help='ID of existing sweep to run')
    parser.add_argument('--count', type=int, default=20, help='Number of runs to perform')
    parser.add_argument('--project', type=str, default='DL_Assign2', help='Wandb project name')
    parser.add_argument('--entity', type=str, default="ch21b053-indian-institute-of-technology-madras", help='Wandb entity name')
    
    args = parser.parse_args()
    
    if args.create:
        # Create a new sweep
        sweep_config = create_sweep_configuration()
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep with ID: {sweep_id}")
        print(f"To run the sweep, use: python sweep.py --sweep_id {sweep_id} --count {args.count}")
    
    elif args.sweep_id:
        # Run an existing sweep
        wandb.agent(args.sweep_id, function=run_sweep_agent(), count=args.count, project=args.project, entity=args.entity)
    
    else:
        print("Error: Must either create a new sweep with --create or specify an existing sweep with --sweep_id")

if __name__ == "__main__":
    main()
