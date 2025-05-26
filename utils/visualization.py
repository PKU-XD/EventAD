import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from dagr.utils.buffers import format_data

def validate_and_visualize(model, val_loader, device, result_dir, epoch):
    """Validate model and generate visualization results
    
    Args:
        model: Model
        val_loader: Validation data loader
        device: Computation device
        result_dir: Result save directory
        epoch: Current epoch
        
    Returns:
        tuple: (avg_val_loss, roc_auc, ap)
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    valid_batch_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Validation")):
            # Move data to device
            data = data.to(device)
            data = format_data(data)
            
            # Check bounding box data
            if not hasattr(data, 'bbox') or data.bbox is None or (isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] == 0):
                print(f"Validation batch {i} has no valid bounding boxes, skipping")
                continue
                
            # Get labels
            labels = data.bbox[:, 4]  # 5th column is class label
            
            # Forward pass
            losses, batch_outputs, batch_labels = model(data, labels, testing=True)
            
            # Ensure losses dictionary contains 'cross_entropy' key
            if 'cross_entropy' not in losses:
                print(f"Losses dictionary in validation batch {i} does not contain 'cross_entropy' key, skipping")
                continue
            
            # Accumulate loss
            val_loss += losses['cross_entropy'].item()
            valid_batch_count += 1
            
            # Collect predictions and labels
            for frame_outputs, frame_labels in zip(batch_outputs, batch_labels):
                for output, label in zip(frame_outputs, frame_labels):
                    # Properly handle different output tensor shapes
                    if output.dim() == 1:  # If it's a 1D tensor [2]
                        pred = output[1]
                    elif output.dim() == 2:  # If it's a 2D tensor [1, 2]
                        pred = output[0, 1]  # Use multi-dimensional indexing
                    else:
                        print(f"Warning: Unexpected output dimensions: {output.dim()}, shape: {output.shape}")
                        continue
                    
                    all_preds.append(pred.item())
                    all_labels.append(label.item())
    
    # If no valid batches, raise exception
    if valid_batch_count == 0:
        raise RuntimeError("No valid batches during validation, please check data")
        
    # Calculate average validation loss
    avg_val_loss = val_loss / valid_batch_count
    
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
    plot_roc_curve(fpr, tpr, roc_auc, result_dir, epoch)
    
    # Plot PR curve
    plot_pr_curve(recall, precision, ap, result_dir, epoch)
    
    print(f"Validation results - Loss: {avg_val_loss:.4f}, ROC AUC: {roc_auc:.4f}, AP: {ap:.4f}")
    
    return avg_val_loss, roc_auc, ap

def plot_roc_curve(fpr, tpr, roc_auc, result_dir, epoch):
    """Plot ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under ROC curve
        result_dir: Result save directory
        epoch: Current epoch
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f'roc_curve_epoch_{epoch}.png'))
    plt.close()

def plot_pr_curve(recall, precision, ap, result_dir, epoch):
    """Plot PR curve
    
    Args:
        recall: Recall
        precision: Precision
        ap: Average precision
        result_dir: Result save directory
        epoch: Current epoch
    """
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - Epoch {epoch}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(result_dir, f'pr_curve_epoch_{epoch}.png'))
    plt.close()