import os
import time
import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path

from models.EventAD import EventADModel
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from config.eventad_config import parse_eventad_args
from dagr.utils.buffers import format_data

class EventADTester:
    def __init__(self, args):
        """
        EventAD model tester
        
        Args:
            args: Configuration parameters
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create necessary directories
        self.setup_directories()
        
        # Load test data
        self.setup_test_loader()
        
        # Load DAGR model
        self.setup_dagr_model()
        
        # Initialize model
        self.setup_model()
        
        # Create result recording file
        self.setup_result_file()

    def setup_directories(self):
        """Create necessary directories"""
        # Create unique test result directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(self.args.output_dir, "test_results", f"{self.args.experiment_name}_{timestamp}")
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Create visualization directory
        self.vis_dir = os.path.join(self.result_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

    def setup_test_loader(self):
        """Set up test data loader"""
        print("Loading test dataset...")
        
        # Load test dataset
        self.test_dataset = DSEC(
            self.args,
            self.args.dataset_directory, 
            "test",
            Augmentations.transform_testing,
            debug=False, 
            min_bbox_diag=15, 
            min_bbox_height=10
        )
        
        # Create test data loader
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.args.batch_size,
            shuffle=False,
            follow_batch=['bbox', 'bbox0'],
            num_workers=self.args.num_workers
        )
        
        print(f"Test data: {len(self.test_dataset)} samples")

    def setup_dagr_model(self):
        """Load DAGR model"""
        print("Loading DAGR model...")
        
        # Create DAGR model
        self.dagr_model = DAGR(
            self.args, 
            height=self.test_dataset.height, 
            width=self.test_dataset.width
        ).to(self.device)
        
        # Create EMA model instance
        self.ema = ModelEMA(self.dagr_model)
        
        # Load pretrained weights
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        self.ema.ema.load_state_dict(checkpoint['ema'])
        
        # Cache lookup tables
        self.ema.ema.cache_luts(
            radius=self.args.radius, 
            height=self.test_dataset.height, 
            width=self.test_dataset.width
        )
        
        print("DAGR model loaded")

    def setup_model(self):
        """Initialize model and load best checkpoint"""
        # Create model instance
        self.model = EventADModel(
            dagr_model=self.ema.ema,
            x_dim=self.args.x_dim,
            h_dim=self.args.h_dim,
            n_frames=self.args.n_frames,
            fps=self.args.fps
        ).to(self.device)
        
        # Determine checkpoint to load
        # Prioritize checkpoint specified in command line
        if hasattr(self.args, 'test_checkpoint') and self.args.test_checkpoint:
            checkpoint_path = self.args.test_checkpoint
        else:
            # Otherwise try to load the best AP model
            model_dir = Path(self.args.output_dir) / "models"
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
            
            # Find the latest experiment directory
            experiment_dirs = sorted(list(model_dir.glob(f"{self.args.experiment_name}_*")), reverse=True)
            if not experiment_dirs:
                raise FileNotFoundError(f"No directories matching experiment name: {self.args.experiment_name}")
            
            latest_exp_dir = experiment_dirs[0]
            
            # Prioritize best AP model
            best_ap_path = latest_exp_dir / "best_ap_model.pth"
            best_auc_path = latest_exp_dir / "best_auc_model.pth"
            latest_path = latest_exp_dir / "latest_checkpoint.pth"
            
            if best_ap_path.exists():
                checkpoint_path = best_ap_path
            elif best_auc_path.exists():
                checkpoint_path = best_auc_path
            elif latest_path.exists():
                checkpoint_path = latest_path
            else:
                raise FileNotFoundError(f"No checkpoints found in directory {latest_exp_dir}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Print model information
        print(f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Record loaded checkpoint information
        self.checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_auc': checkpoint.get('best_auc', 'unknown'),
            'best_ap': checkpoint.get('best_ap', 'unknown')
        }

    def setup_result_file(self):
        """Create result recording file"""
        self.result_file = os.path.join(self.result_dir, 'test_results.csv')
        
        with open(self.result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment', self.args.experiment_name])
            writer.writerow(['Dataset', self.args.dataset_directory])
            writer.writerow(['Model Parameters', f"x_dim: {self.args.x_dim}, h_dim: {self.args.h_dim}"])
            writer.writerow(['Checkpoint', self.checkpoint_info['path']])
            writer.writerow(['Epoch', self.checkpoint_info['epoch']])
            writer.writerow(['Best AUC', self.checkpoint_info['best_auc']])
            writer.writerow(['Best AP', self.checkpoint_info['best_ap']])
            writer.writerow([])  # Empty row
            writer.writerow(['Metric', 'Value'])

    def test(self):
        """Test model and generate evaluation report"""
        print("Starting model testing...")
        
        all_preds = []
        all_labels = []
        all_scores = []  # Store raw prediction scores
        sample_ids = []  # Store sample IDs for later analysis
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader, desc="Testing progress")):
                try:
                    # Move data to device
                    data = data.to(self.device)
                    data = format_data(data)
                    
                    # Get sample IDs
                    if hasattr(data, 'sample_id'):
                        batch_ids = data.sample_id
                    else:
                        # If no sample ID, use batch index
                        batch_ids = [f"batch_{i}_sample_{j}" for j in range(len(data.x))]
                    
                    # Get labels
                    if hasattr(data, 'bbox') and data.bbox is not None:
                        if isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] > 0:
                            labels = data.bbox[:, 4]  # Assume 5th column is class label
                        else:
                            print(f"Batch {i} has no valid bounding boxes, skipping")
                            continue
                    else:
                        print(f"Batch {i} has no bbox attribute, skipping")
                        continue
                    
                    # Forward pass
                    losses, batch_outputs, batch_labels = self.model(data, labels, testing=True)
                    
                    # Collect predictions and labels
                    for j, (frame_outputs, frame_labels) in enumerate(zip(batch_outputs, batch_labels)):
                        for k, (output, label) in enumerate(zip(frame_outputs, frame_labels)):
                            # Extract anomaly class prediction probability
                            score = output[1]  # Assume index 1 corresponds to anomaly class raw score
                            pred = 1 if score > 0.5 else 0  # Use 0.5 as default threshold
                            
                            all_preds.append(pred)
                            all_labels.append(label.item())
                            all_scores.append(score.item())
                            
                            # If batch_ids is a list, ensure index is valid
                            if isinstance(batch_ids, list) and j < len(batch_ids):
                                sample_ids.append(f"{batch_ids[j]}_{k}")
                            else:
                                sample_ids.append(f"batch_{i}_frame_{j}_obj_{k}")
                
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        # Calculate various evaluation metrics
        self.calculate_metrics(all_labels, all_preds, all_scores)
        
        # Generate detailed sample-level prediction results
        self.save_detailed_results(sample_ids, all_labels, all_preds, all_scores)
        
        print(f"Testing completed! Results saved in: {self.result_dir}")

    def calculate_metrics(self, labels, preds, scores):
        """Calculate and save various evaluation metrics"""
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve and AP
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Calculate classification report
        report = classification_report(labels, preds, target_names=['Normal', 'Anomaly'], output_dict=True)
        
        # Find threshold with best F1 score
        f1_scores = []
        for threshold in np.arange(0.1, 1.0, 0.05):
            threshold_preds = (scores >= threshold).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, threshold_preds)
            f1_scores.append((threshold, f1))
        
        best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
        
        # Recalculate predictions using best threshold
        best_preds = (scores >= best_threshold).astype(int)
        best_cm = confusion_matrix(labels, best_preds)
        best_report = classification_report(labels, best_preds, target_names=['Normal', 'Anomaly'], output_dict=True)
        
        # Save results to CSV
        with open(self.result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ROC AUC', f"{roc_auc:.4f}"])
            writer.writerow(['AP', f"{ap:.4f}"])
            writer.writerow(['Best Threshold', f"{best_threshold:.4f}"])
            writer.writerow(['Best F1 Score', f"{best_f1:.4f}"])
            writer.writerow(['Accuracy', f"{report['accuracy']:.4f}"])
            writer.writerow(['Anomaly Precision', f"{report['Anomaly']['precision']:.4f}"])
            writer.writerow(['Anomaly Recall', f"{report['Anomaly']['recall']:.4f}"])
            writer.writerow(['Anomaly F1 Score', f"{report['Anomaly']['f1-score']:.4f}"])
            writer.writerow(['Best Threshold Anomaly Precision', f"{best_report['Anomaly']['precision']:.4f}"])
            writer.writerow(['Best Threshold Anomaly Recall', f"{best_report['Anomaly']['recall']:.4f}"])
            writer.writerow(['Best Threshold Anomaly F1 Score', f"{best_report['Anomaly']['f1-score']:.4f}"])
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.vis_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR curve (AP = {ap:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.vis_dir, 'pr_curve.png'))
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Threshold=0.5)')
        plt.savefig(os.path.join(self.vis_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot confusion matrix with best threshold
        plt.figure(figsize=(8, 6))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Best Threshold={best_threshold:.2f})')
        plt.savefig(os.path.join(self.vis_dir, 'best_threshold_confusion_matrix.png'))
        plt.close()
        
        # Plot score distribution
        plt.figure(figsize=(12, 6))
        
        # Plot score distribution for normal and anomaly samples separately
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal Samples')
        plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly Samples')
        
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold=0.5')
        plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best Threshold={best_threshold:.2f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Sample Count')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.savefig(os.path.join(self.vis_dir, 'score_distribution.png'))
        plt.close()
        
        # Plot F1 score vs threshold
        thresholds = [t for t, _ in f1_scores]
        f1s = [f for _, f in f1_scores]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1s, marker='o')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold={best_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.vis_dir, 'f1_vs_threshold.png'))
        plt.close()
        
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}, F1: {best_f1:.4f}")
        print(f"Classification Report:\n{classification_report(labels, preds, target_names=['Normal', 'Anomaly'])}")

    def save_detailed_results(self, sample_ids, labels, preds, scores):
        """Save detailed sample-level prediction results"""
        # Create detailed results CSV file
        detailed_file = os.path.join(self.result_dir, 'detailed_predictions.csv')
        
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample ID', 'True Label', 'Predicted Label', 'Prediction Score', 'Correct Prediction'])
            
            for sample_id, label, pred, score in zip(sample_ids, labels, preds, scores):
                correct = 1 if pred == label else 0
                writer.writerow([sample_id, label, pred, f"{score:.4f}", correct])
        
        # Find misclassified samples
        false_positives = [(id, score) for id, label, pred, score in zip(sample_ids, labels, preds, scores) 
                           if label == 0 and pred == 1]
        
        false_negatives = [(id, score) for id, label, pred, score in zip(sample_ids, labels, preds, scores) 
                           if label == 1 and pred == 0]
        
        # Save misclassified sample information
        error_file = os.path.join(self.result_dir, 'misclassified_samples.csv')
        
        with open(error_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Error Type', 'Sample ID', 'Prediction Score'])
            
            for id, score in false_positives:
                writer.writerow(['False Positive (FP)', id, f"{score:.4f}"])
            
            for id, score in false_negatives:
                writer.writerow(['False Negative (FN)', id, f"{score:.4f}"])
        
        print(f"False Positive samples count: {len(false_positives)}")
        print(f"False Negative samples count: {len(false_negatives)}")


def add_test_args(parser):
    """Add test-specific command line arguments"""
    parser.add_argument('--test_checkpoint', type=str, default='', 
                        help='Path to specific checkpoint to test, if not specified will use best AP model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold, default is 0.5')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Whether to save predictions for each sample')
    parser.add_argument('--visualize_samples', type=int, default=0,
                        help='Number of samples to visualize, 0 means no visualization')
    return parser


if __name__ == '__main__':
    # Use configuration parsing function to get parameters
    args = parse_eventad_args()
    
    # Add test-specific parameters
    import argparse
    parser = argparse.ArgumentParser(description='EventAD Model Testing')
    parser = add_test_args(parser)
    test_args, _ = parser.parse_known_args()
    
    # Add test-specific parameters to args
    for arg in vars(test_args):
        setattr(args, arg, getattr(test_args, arg))
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create and start tester
    tester = EventADTester(args)
    tester.test()