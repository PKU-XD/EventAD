import os
import torch
from models.EventAD import EventADModel
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

def setup_dagr_model(args, val_dataset, device):
    """Load DAGR model
    
    Args:
        args: Configuration parameters
        val_dataset: Validation dataset, used to get height and width
        device: Computation device
        
    Returns:
        ModelEMA: EMA model instance
    """
    print("Loading DAGR model...")
    
    # Create DAGR model
    dagr_model = DAGR(
        args, 
        height=val_dataset.height, 
        width=val_dataset.width
    ).to(device)
    
    # Create EMA model instance
    ema = ModelEMA(dagr_model)
    
    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ema.ema.load_state_dict(checkpoint['ema'])
    
    # Cache lookup tables
    ema.ema.cache_luts(
        radius=args.radius, 
        height=val_dataset.height, 
        width=val_dataset.width
    )
    
    print("DAGR model loading completed")
    return ema

def setup_model(args, ema, device):
    """Initialize EventAD model
    
    Args:
        args: Configuration parameters
        ema: EMA model instance
        device: Computation device
        
    Returns:
        EventADModel: Initialized model
    """
    # Create model instance - using EMA model
    model = EventADModel(
        dagr_model=ema.ema,  # Use EMA model
        x_dim=args.x_dim, 
        h_dim=args.h_dim,
        n_frames=args.n_frames,
        fps=args.fps
    ).to(device)
    
    # Print model structure
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        checkpoint_path: Checkpoint path
        device: Computation device
        
    Returns:
        tuple: (start_epoch, best_auc, best_ap)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist")
        return 0, 0, 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model parameters
    model.load_state_dict(checkpoint['model'])
    
    # Also load optimizer state if included
    if 'optimizer' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update starting epoch if epoch information is included
    start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0
    best_auc = checkpoint['best_auc'] if 'best_auc' in checkpoint else 0
    best_ap = checkpoint['best_ap'] if 'best_ap' in checkpoint else 0
    
    print(f"Loaded checkpoint from {checkpoint_path}, continuing training from epoch {start_epoch}")
    return start_epoch, best_auc, best_ap

def save_checkpoint(model, optimizer, epoch, best_auc, best_ap, model_dir, is_best_auc=False, is_best_ap=False):
    """Save checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        best_auc: Best AUC value
        best_ap: Best AP value
        model_dir: Model save directory
        is_best_auc: Whether this is the best AUC model
        is_best_ap: Whether this is the best AP model
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_auc': best_auc,
        'best_ap': best_ap
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(model_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save separately if it's the best AUC model
    if is_best_auc:
        best_auc_path = os.path.join(model_dir, 'best_auc_model.pth')
        torch.save(checkpoint, best_auc_path)
        print(f"Saved best AUC model: {best_auc_path}")
    
    # Save separately if it's the best AP model
    if is_best_ap:
        best_ap_path = os.path.join(model_dir, 'best_ap_model.pth')
        torch.save(checkpoint, best_ap_path)
        print(f"Saved best AP model: {best_ap_path}")