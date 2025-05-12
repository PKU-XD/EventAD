import os
import time
from datetime import datetime
import numpy as np
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from models.EventAD import EventADModel
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from config.eventad_config import parse_eventad_args
from dagr.utils.buffers import format_data

class EventADTrainer:
    def __init__(self, args):
        """
        EventAD model trainer
        
        Args:
            args: Configuration parameters
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create necessary directories
        self.setup_directories()
        
        # Create data augmentation object
        self.setup_augmentations()

        # Load data
        self.setup_data_loaders()

        # Load DAGR model
        self.setup_dagr_model()
        
        # Initialize model
        self.setup_model()
        
        # Set up optimizer and learning rate scheduler
        self.setup_optimizer()
        
        # Record training information
        self.best_auc = 0
        self.best_ap = 0
        self.start_epoch = 0
        
        # Create result recording file
        self.setup_result_file()

    def setup_augmentations(self):
        """Create data augmentation object"""
        # Create Augmentations instance to get transform_training
        self.augmentations = Augmentations(self.args)        


    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = os.path.join(self.args.output_dir, "models", f"{self.args.experiment_name}_{timestamp}")
        self.result_dir = os.path.join(self.args.output_dir, "results", f"{self.args.experiment_name}_{timestamp}")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        

    def setup_data_loaders(self):
        """Set up data loaders"""
        print("Loading datasets...")
        
        # Load training and validation datasets
        self.train_dataset = DSEC(
            self.args,
            self.args.dataset_directory, 
            "train", 
            self.augmentations.transform_testing,
            debug=False, 
            min_bbox_diag=15, 
            min_bbox_height=10
        )
        
        self.val_dataset = DSEC(
            self.args,
            self.args.dataset_directory, 
            "val", 
            Augmentations.transform_testing,
            debug=False, 
            min_bbox_diag=15, 
            min_bbox_height=10
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True, 
            follow_batch=['bbox', 'bbox0'],
            num_workers=self.args.num_workers
        )
        
        self.val_loader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.args.batch_size,
            shuffle=False, 
            follow_batch=['bbox', 'bbox0'],
            num_workers=self.args.num_workers
        )
        
        print(f"Training data: {len(self.train_dataset)} samples, Validation data: {len(self.val_dataset)} samples")
    

    def setup_dagr_model(self):
        """Load DAGR model"""
        print("Loading DAGR model...")
        
        # Create DAGR model
        self.dagr_model = DAGR(
            self.args, 
            height=self.val_dataset.height, 
            width=self.val_dataset.width
        ).to(self.device)
        
        # Create EMA model instance
        self.ema = ModelEMA(self.dagr_model)
        
        # Load pretrained weights
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        self.ema.ema.load_state_dict(checkpoint['ema'])
        
        # Cache lookup tables
        self.ema.ema.cache_luts(
            radius=self.args.radius, 
            height=self.val_dataset.height, 
            width=self.val_dataset.width
        )
        
        print("DAGR model loaded")


    def setup_model(self):
        """Initialize model"""
        # Create model instance - use EMA model
        self.model = EventADModel(
            dagr_model=self.ema.ema,
            x_dim=self.args.x_dim, 
            h_dim=self.args.h_dim,
            n_frames=self.args.n_frames,
            fps=self.args.fps
        ).to(self.device)
        
        # Load pretrained model if specified
        if self.args.pretrained_model:
            self._load_checkpoint(self.args.pretrained_model)
            
        # Print model structure
        print(f"Total model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_optimizer(self):
        """Set up optimizer and learning rate scheduler"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') and self.args.weight_decay is not None else 1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def setup_result_file(self):
        """Create result recording file"""
        self.result_file = os.path.join(self.result_dir, 'training_results.csv')
        
        with open(self.result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment', self.args.experiment_name])
            writer.writerow(['Dataset', self.args.dataset_directory])
            writer.writerow(['Model Parameters', f"x_dim: {self.args.x_dim}, h_dim: {self.args.h_dim}"])
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'ROC AUC', 'AP', 'Learning Rate'])
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model'])
        
        # Also load optimizer state if included
        if 'optimizer' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Update starting epoch if epoch info is included
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            
        print(f"Loaded checkpoint from {checkpoint_path}, continuing training from epoch {self.start_epoch}")
    
    def save_checkpoint(self, epoch, is_best_auc=False, is_best_ap=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_ap': self.best_ap
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # If best AUC model, save separately
        if is_best_auc:
            best_auc_path = os.path.join(self.model_dir, 'best_auc_model.pth')
            torch.save(checkpoint, best_auc_path)
            print(f"Saved best AUC model: {best_auc_path}")
        
        # If best AP model, save separately
        if is_best_ap:
            best_ap_path = os.path.join(self.model_dir, 'best_ap_model.pth')
            torch.save(checkpoint, best_ap_path)
            print(f"Saved best AP model: {best_ap_path}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0
        nan_count = 0
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}")
        
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in model: {trainable_params}")
        
        # Check main components of the model
        print(f"Model structure: {self.model}")
        
        # Process only a few batches for diagnostics
        max_debug_batches = 3
        debug_batch_count = 0
        
        for i, data in progress_bar:
            try:
                # Add debug information
                if debug_batch_count < max_debug_batches:
                    print(f"\n===== Debug batch {i} =====")
                    print(f"Data type: {type(data)}")
                    print(f"Data attributes: {data.keys()}")
                    if hasattr(data, 'bbox'):
                        print(f"Bounding box shape: {data.bbox.shape if isinstance(data.bbox, torch.Tensor) else 'Not a tensor'}")
                    
                    # Check if DAGR model is frozen
                    dagr_params_frozen = all(not p.requires_grad for p in self.model.dagr_model.parameters())
                    print(f"DAGR model parameters frozen: {dagr_params_frozen}")
                    
                    debug_batch_count += 1
                
                # Move data to device
                data = data.to(self.device)
                data = format_data(data)
                
                # Get labels
                if hasattr(data, 'bbox') and data.bbox is not None:
                    if isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] > 0:
                        labels = data.bbox[:, 4]  # Assume 5th column is class label
                        if debug_batch_count <= max_debug_batches:
                            print(f"Labels shape: {labels.shape}, values: {labels[:10]}")
                    else:
                        print(f"Batch {i} has no valid bounding boxes, skipping")
                        continue
                else:
                    print(f"Batch {i} has no bbox attribute, skipping")
                    continue
                
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                try:
                    if debug_batch_count <= max_debug_batches:
                        print("Executing forward pass...")
                    
                    # Try to track errors in forward pass
                    with torch.autograd.detect_anomaly():
                        losses, outputs, _ = self.model(data, labels)
                    
                    if debug_batch_count <= max_debug_batches:
                        print(f"Forward pass completed, losses: {losses}")
                        if isinstance(outputs, list) and len(outputs) > 0:
                            print(f"Output shape: {len(outputs)} batches, each batch contains {len(outputs[0]) if len(outputs) > 0 else 0} frames")
                    
                    # Ensure losses dictionary contains 'cross_entropy' key
                    if 'cross_entropy' not in losses:
                        print(f"Losses dictionary in batch {i} does not contain 'cross_entropy' key, skipping")
                        nan_count += 1
                        continue
                    
                    loss = losses['cross_entropy']
                    
                    # Check if loss is scalar
                    if not loss.dim() == 0:
                        print(f"Loss in batch {i} is not a scalar but has shape {loss.shape}, trying to take mean")
                        loss = loss.mean()
                    
                    # Check if loss contains NaN or inf
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"Loss in batch {i} contains NaN or inf values, skipping")
                        nan_count += 1
                        continue
                    
                    # Check if loss has gradient
                    if not loss.requires_grad:
                        print(f"Loss in batch {i} has no gradient, checking computation graph...")
                        # Print loss computation graph info
                        print(f"Loss gradient function: {loss.grad_fn}")
                        
                        # Check if model parameters require gradient
                        requires_grad_params = [(name, param.requires_grad) for name, param in self.model.named_parameters()]
                        print(f"Model parameter gradient status: {requires_grad_params[:5]}... (total {len(requires_grad_params)})")
                        
                        # Try to recompute loss
                        try:
                            with torch.enable_grad():
                                # Ensure model is in training mode
                                self.model.train()
                                # Recompute loss
                                losses, outputs, labels_out = self.model(data, labels)
                                loss = losses['cross_entropy']
                                
                                # If loss still has no gradient, skip this batch
                                if not loss.requires_grad:
                                    print(f"Loss in batch {i} still has no gradient, skipping")
                                    nan_count += 1
                                    continue
                        except Exception as e:
                            print(f"Error recomputing loss: {e}")
                            import traceback
                            traceback.print_exc()
                            nan_count += 1
                            continue
                        
                    # Backward pass
                    if debug_batch_count <= max_debug_batches:
                        print("Executing backward pass...")
                    
                    loss.backward()
                    
                    if debug_batch_count <= max_debug_batches:
                        print("Backward pass completed")
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    # Check if gradients contain NaN or inf
                    has_bad_grad = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Gradient of parameter {name} contains NaN or inf values")
                                has_bad_grad = True
                                break
                    
                    if has_bad_grad:
                        nan_count += 1
                        continue
                        
                    # Update parameters
                    self.optimizer.step()
                    
                    # Update progress bar
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({
                        'loss': loss.item(), 
                        'avg_loss': epoch_loss / (i + 1 - nan_count),
                        'nan_count': nan_count,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                
                except RuntimeError as e:
                    print(f"Runtime error: {e}")
                    import traceback
                    traceback.print_exc()
                    nan_count += 1
                    continue
                except Exception as e:
                    print(f"Error during forward or backward pass: {e}")
                    import traceback
                    traceback.print_exc()
                    nan_count += 1
                    continue
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                nan_count += 1
                continue
        
        # Calculate average loss
        valid_batches = len(self.train_loader) - nan_count
        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
            print(f"Epoch {epoch} training completed, average loss: {avg_epoch_loss:.4f}, valid batches: {valid_batches}/{len(self.train_loader)}")
        else:
            print(f"Warning: Epoch {epoch} has no valid batches! All batches were skipped.")
            avg_epoch_loss = float('nan')
        
        return avg_epoch_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                data = data.to(self.device)
                
                # Use format_data to process data
                from dagr.utils.buffers import format_data
                data = format_data(data)
                
                # Get labels
                labels = data.bbox[:, 4]  # Assume 5th column is class label
                
                # Forward pass
                losses, batch_outputs, batch_labels = self.model(data, labels, testing=True)
                
                # Accumulate loss
                val_loss += losses['cross_entropy'].item()
                
                # Collect predictions and labels
                for frame_outputs, frame_labels in zip(batch_outputs, batch_labels):
                    for output, label in zip(frame_outputs, frame_labels):
                        # Extract anomaly class prediction probability
                        pred = output[1]  # Assume index 1 corresponds to anomaly class
                        all_preds.append(pred)
                        all_labels.append(label)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve and AP
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.result_dir, f'roc_curve_epoch_{epoch}.png'))
        plt.close()
        
        # Plot PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR curve (AP = {ap:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - Epoch {epoch}')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.result_dir, f'pr_curve_epoch_{epoch}.png'))
        plt.close()
        
        print(f"Validation results - Loss: {avg_val_loss:.4f}, ROC AUC: {roc_auc:.4f}, AP: {ap:.4f}")
        
        return avg_val_loss, roc_auc, ap
    
    def train(self):
        """Train model"""
        print(f"Starting training - Total {self.args.epochs} epochs")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\n===== Epoch {epoch}/{self.args.epochs-1} =====")
            
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, roc_auc, ap = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save results to CSV
            with open(self.result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, roc_auc, ap, current_lr])
            
            # Check if this is the best model
            is_best_auc = False
            is_best_ap = False
            
            if roc_auc > self.best_auc:
                self.best_auc = roc_auc
                is_best_auc = True
                print(f"New best AUC: {self.best_auc:.4f}")
            
            if ap > self.best_ap:
                self.best_ap = ap
                is_best_ap = True
                print(f"New best AP: {self.best_ap:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best_auc, is_best_ap)
            
            # Early stopping if learning rate is too small
            if current_lr < self.args.min_lr:
                print(f"Learning rate ({current_lr}) below threshold ({self.args.min_lr}), stopping training early")
                break
        
        print(f"Training completed! Best AUC: {self.best_auc:.4f}, Best AP: {self.best_ap:.4f}")
        print(f"Models saved in: {self.model_dir}")
        print(f"Results saved in: {self.result_dir}")


if __name__ == '__main__':
    # Use our configuration parsing function to get parameters
    args = parse_eventad_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create and start trainer
    trainer = EventADTrainer(args)
    trainer.train()