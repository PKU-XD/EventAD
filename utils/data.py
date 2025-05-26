import torch
from torch_geometric.data import DataLoader
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations

def setup_augmentations(args):
    """Create data augmentation object"""
    # Create Augmentations instance to get transform_training
    return Augmentations(args)

def setup_data_loaders(args, augmentations):
    """Setup data loaders
    
    Args:
        args: Configuration parameters
        augmentations: Data augmentation object
        
    Returns:
        tuple: (train_dataset, val_dataset, train_loader, val_loader)
    """
    print("Loading datasets...")
    
    # Load training and validation datasets
    train_dataset = DSEC(
        args,
        args.dataset_directory, 
        "test",  # Note: Using "test" as training set, may need adjustment based on actual situation
        # "val",
        # augmentations.transform_training,
        augmentations.transform_testing,
        debug=False, 
        min_bbox_diag=15, 
        min_bbox_height=10
    )
    
    val_dataset = DSEC(
        args,
        args.dataset_directory, 
        "val", 
        Augmentations.transform_testing,  # This is a class attribute, can be used directly
        debug=False, 
        min_bbox_diag=15, 
        min_bbox_height=10
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        follow_batch=['bbox', 'bbox0'],
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        follow_batch=['bbox', 'bbox0'],
        num_workers=args.num_workers
    )
    
    print(f"Training data: {len(train_dataset)} samples, Validation data: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset, train_loader, val_loader

def check_dataset_balance(train_loader, val_loader):
    """Check class distribution in datasets
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    print("Checking dataset class distribution...")
    
    # Check training set
    train_labels = []
    for data in train_loader:
        if hasattr(data, 'bbox') and data.bbox is not None:
            labels = data.bbox[:, 4]
            train_labels.extend(labels.cpu().numpy())
    
    train_pos = sum(1 for l in train_labels if l > 0.5)
    train_neg = len(train_labels) - train_pos
    print(f"Training set: Total samples {len(train_labels)}, Normal samples {train_neg}, Anomaly samples {train_pos}, Anomaly ratio {train_pos/len(train_labels)*100:.2f}%")
    
    # Check validation set
    val_labels = []
    for data in val_loader:
        if hasattr(data, 'bbox') and data.bbox is not None:
            labels = data.bbox[:, 4]
            val_labels.extend(labels.cpu().numpy())
    
    val_pos = sum(1 for l in val_labels if l > 0.5)
    val_neg = len(val_labels) - val_pos
    print(f"Validation set: Total samples {len(val_labels)}, Normal samples {val_neg}, Anomaly samples {val_pos}, Anomaly ratio {val_pos/len(val_labels)*100:.2f}%")