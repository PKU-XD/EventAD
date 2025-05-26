import os
import csv
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from dagr.utils.buffers import format_data

def setup_directories(args):
    """Create necessary directories
    
    Args:
        args: Configuration parameters
        
    Returns:
        tuple: (model_dir, result_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, "models", f"{args.experiment_name}_{timestamp}")
    result_dir = os.path.join(args.output_dir, "results", f"{args.experiment_name}_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    return model_dir, result_dir

def setup_optimizer(model, args):
    """Setup optimizer and learning rate scheduler
    
    Args:
        model: Model
        args: Configuration parameters
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') and args.weight_decay is not None else 1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return optimizer, scheduler

def setup_result_file(result_dir, args):
    """Create result recording file
    
    Args:
        result_dir: Result directory
        args: Configuration parameters
        
    Returns:
        str: Result file path
    """
    result_file = os.path.join(result_dir, 'training_results.csv')
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', args.experiment_name])
        writer.writerow(['Dataset', args.dataset_directory])
        writer.writerow(['Model Parameters', f"x_dim: {args.x_dim}, h_dim: {args.h_dim}"])
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'ROC AUC', 'AP', 'Learning Rate'])
    
    return result_file

def train_epoch(model, train_loader, optimizer, device, args, epoch):
    """Train one epoch
    
    Args:
        model: Model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Computation device
        args: Configuration parameters
        epoch: Current epoch
        
    Returns:
        float: Average training loss
    """
    model.train()
    epoch_loss = 0
    valid_batch_count = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    # Debug first few batches
    debug_mode = epoch == 0
    max_debug_batches = 3 if debug_mode else 0
    
    for i, data in progress_bar:
        # Move data to device
        data = data.to(device)
        data = format_data(data)
        
        # Check bounding box data
        if not hasattr(data, 'bbox') or data.bbox is None or (isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] == 0):
            print(f"Batch {i} has no valid bounding boxes, skipping")
            continue
            
        # Get labels
        labels = data.bbox[:, 4]  # 5th column is class label
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Use autograd.detect_anomaly to detect errors in computation graph
        with torch.autograd.detect_anomaly():
            # Forward pass
            losses, outputs, _ = model(data, labels)
            
            # Ensure losses dictionary contains 'cross_entropy' key
            if 'cross_entropy' not in losses:
                raise ValueError(f"Losses dictionary in batch {i} does not contain 'cross_entropy' key")
            
            loss = losses['cross_entropy']
            
            # Check if loss is a scalar
            if not loss.dim() == 0:
                loss = loss.mean()
            
            # Check if loss contains NaN or inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError(f"Loss in batch {i} contains NaN or inf values: {loss.item()}")
            
            # Check if loss has gradient
            if not loss.requires_grad:
                raise ValueError(f"Loss in batch {i} has no gradient")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Check if gradients contain NaN or inf
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    raise ValueError(f"Gradient of parameter {name} contains NaN or inf values")
            
            # Update parameters
            optimizer.step()
        
        # Update statistics
        epoch_loss += loss.item()
        valid_batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(), 
            'avg_loss': epoch_loss / valid_batch_count if valid_batch_count > 0 else float('nan'),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Calculate average loss
    if valid_batch_count > 0:
        avg_epoch_loss = epoch_loss / valid_batch_count
        print(f"Epoch {epoch} training completed, average loss: {avg_epoch_loss:.4f}, valid batches: {valid_batch_count}/{len(train_loader)}")
        return avg_epoch_loss
    else:
        print(f"Warning: Epoch {epoch} has no valid batches! All batches were skipped.")
        # Raise exception if no valid batches
        raise RuntimeError("No valid batches during training, please check data and model")