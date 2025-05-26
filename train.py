# -*- coding: utf-8 -*-
import numpy as np
import torch
import csv
import argparse

# Import custom modules
from utils.data import setup_augmentations, setup_data_loaders, check_dataset_balance
from utils.model import setup_dagr_model, setup_model, load_checkpoint, save_checkpoint
from utils.train import setup_directories, setup_optimizer, setup_result_file, train_epoch
from utils.visualization import validate_and_visualize
from config.eventad_config import parse_eventad_args

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def add_train_args(parser):
    """Add training-specific command line arguments"""
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay coefficient')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epoch interval for saving checkpoints')
    parser.add_argument('--plot_interval', type=int, default=5,
                        help='Epoch interval for plotting curves')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='Learning rate decay factor')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience value for learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume training')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold, default is 0.5')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Pretrained model path')
    
    return parser

def train(args):
    """Main training function
    
    Args:
        args: Configuration parameters
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    model_dir, result_dir = setup_directories(args)
    
    # Create data augmentation objects
    augmentations = setup_augmentations(args)
    
    # Load data
    train_dataset, val_dataset, train_loader, val_loader = setup_data_loaders(args, augmentations)
    
    # Check dataset balance
    # check_dataset_balance(train_loader, val_loader)
    
    # Load DAGR model
    ema = setup_dagr_model(args, val_dataset, device)
    
    # Initialize model
    model = setup_model(args, ema, device)
    
    # Setup optimizer and learning rate scheduler
    optimizer, scheduler = setup_optimizer(model, args)
    
    # Create result recording file
    result_file = setup_result_file(result_dir, args)
    
    # Initialize training state
    start_epoch = 0
    best_auc = 0
    best_ap = 0
    
    # Load pretrained model if specified
    if args.pretrained_model:
        start_epoch, best_auc, best_ap = load_checkpoint(model, optimizer, args.pretrained_model, device)
    
    print(f"Starting training - Total {args.epochs} epochs")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n===== Epoch {epoch}/{args.epochs-1} =====")
            
            # Train one epoch
            train_loss = train_epoch(model, train_loader, optimizer, device, args, epoch)
            
            # Validate
            val_loss, roc_auc, ap = validate_and_visualize(model, val_loader, device, result_dir, epoch)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save results to CSV
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, roc_auc, ap, current_lr])
            
            # Check if it's the best model
            is_best_auc = False
            is_best_ap = False
            
            if roc_auc > best_auc:
                best_auc = roc_auc
                is_best_auc = True
                print(f"New best AUC: {best_auc:.4f}")
            
            if ap > best_ap:
                best_ap = ap
                is_best_ap = True
                print(f"New best AP: {best_ap:.4f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, best_auc, best_ap, model_dir, is_best_auc, is_best_ap)
            
            # Early stopping if learning rate is too small
            if current_lr < args.min_lr:
                print(f"Learning rate ({current_lr}) below threshold ({args.min_lr}), stopping training early")
                break
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, best_auc, best_ap, model_dir, False, False)
        raise  # Re-raise exception to ensure program stops
    
    print(f"Training completed! Best AUC: {best_auc:.4f}, Best AP: {best_ap:.4f}")
    print(f"Models saved in: {model_dir}")
    print(f"Results saved in: {result_dir}")

if __name__ == '__main__':
    # Get parameters using configuration parsing function
    args = parse_eventad_args()
    
    # Add training-specific parameters
    parser = argparse.ArgumentParser(description='EventAD Model Training')
    parser = add_train_args(parser)
    train_args, _ = parser.parse_known_args()
    
    # Add training-specific parameters to args
    for arg in vars(train_args):
        setattr(args, arg, getattr(train_args, arg))
    
    # Set random seed to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Start training
    train(args)