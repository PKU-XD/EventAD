import os
import csv
import numpy as np

def setup_result_file(result_dir, args, checkpoint_info=None):
    """
    Create result recording file
    
    Args:
        result_dir: Result directory path
        args: Configuration parameters
        checkpoint_info: Checkpoint information (optional)
        
    Returns:
        str: Result file path
    """
    result_file = os.path.join(result_dir, 'test_results.csv' if checkpoint_info else 'training_results.csv')
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', args.experiment_name])
        writer.writerow(['Dataset', args.dataset_directory])
        writer.writerow(['Model Parameters', f"x_dim: {args.x_dim}, h_dim: {args.h_dim}"])
        
        if checkpoint_info:
            writer.writerow(['Checkpoint', checkpoint_info['path']])
            writer.writerow(['Epoch', checkpoint_info['epoch']])

        else:
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'ROC AUC', 'AP', 'Learning Rate'])
    
    return result_file

def save_metrics(result_file, bbox_metrics, frame_metrics, tta_metrics=None, response_metrics=None):
    """
    Save all evaluation metrics to CSV file
    
    Args:
        result_file: Result file path
        bbox_metrics: Bounding box level metrics
        frame_metrics: Frame level metrics
        tta_metrics: TTA metrics (optional)
        response_metrics: RESPONSE metrics (optional)
    """
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Save bounding box level metrics
        writer.writerow(['AUC', f"{bbox_metrics.get('auc', 'N/A'):.4f}" if not np.isnan(bbox_metrics.get('auc', np.nan)) else "N/A"])
        writer.writerow(['AP', f"{bbox_metrics.get('ap', 'N/A'):.4f}" if not np.isnan(bbox_metrics.get('ap', np.nan)) else "N/A"])
        
        # Save frame level metrics
        writer.writerow(['AUC-Frame', f"{frame_metrics.get('auc_frame', 'N/A'):.4f}" if not np.isnan(frame_metrics.get('auc_frame', np.nan)) else "N/A"])
        
        # Also save TTA metrics if available
        if tta_metrics:
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                key = f'tta_{threshold}'            
            writer.writerow(['mTTA', f"{tta_metrics.get('mtta', 'N/A'):.4f}" if not np.isnan(tta_metrics.get('mtta', np.nan)) else "N/A"])
        
        # Also save RESPONSE metrics if available
        if response_metrics:
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                key = f'response_{threshold}'
            
            writer.writerow(['mRESPONSE', f"{response_metrics.get('mresponse', 'N/A'):.4f}" if not np.isnan(response_metrics.get('mresponse', np.nan)) else "N/A"])

def create_metrics_summary(result_dir, args, bbox_metrics, frame_metrics, tta_metrics=None, response_metrics=None, checkpoint_info=None, fps_results=None):
    """
    Create metrics summary file
    
    Args:
        result_dir: Result directory path
        args: Configuration parameters
        bbox_metrics: Bounding box level metrics
        frame_metrics: Frame level metrics
        tta_metrics: TTA metrics (optional)
        response_metrics: RESPONSE metrics (optional)
        checkpoint_info: Checkpoint information (optional)
        fps_results: FPS measurement results (optional)
    """
    summary_file = os.path.join(result_dir, 'metrics_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Dataset: {args.dataset_directory}\n")
        
        if checkpoint_info:
            f.write(f"Checkpoint: {os.path.basename(checkpoint_info['path'])}\n\n")
        
        f.write("Main Metrics:\n")
        f.write(f"AUC: {bbox_metrics.get('auc', 'N/A'):.4f}\n" if not np.isnan(bbox_metrics.get('auc', np.nan)) else "AUC: N/A\n")
        f.write(f"AP: {bbox_metrics.get('ap', 'N/A'):.4f}\n" if not np.isnan(bbox_metrics.get('ap', np.nan)) else "AP: N/A\n")
        f.write(f"AUC-Frame: {frame_metrics.get('auc_frame', 'N/A'):.4f}\n" if not np.isnan(frame_metrics.get('frame_roc_auc', np.nan)) else "AUC-Frame: N/A\n")
        
        if tta_metrics:
            f.write(f"mTTA: {tta_metrics.get('mtta', 'N/A'):.4f}\n" if not np.isnan(tta_metrics.get('mtta', np.nan)) else "mTTA: N/A\n")
        
        if response_metrics:
            f.write(f"mRESPONSE: {response_metrics.get('mresponse', 'N/A'):.4f}\n" if not np.isnan(response_metrics.get('mresponse', np.nan)) else "mRESPONSE: N/A\n")
        
        if fps_results:
            f.write(f"\nFPS Measurement:\n")
            f.write(f"FPS: {fps_results['fps']:.2f} \n")
        
        # Print all main metrics
        print("\n==== Main Metrics Summary ====")
        print(f"AUC: {bbox_metrics.get('auc', 'N/A'):.4f}" if not np.isnan(bbox_metrics.get('auc', np.nan)) else "AUC: N/A")
        print(f"AP: {bbox_metrics.get('ap', 'N/A'):.4f}" if not np.isnan(bbox_metrics.get('ap', np.nan)) else "AP: N/A")
        print(f"AUC-Frame: {frame_metrics.get('auc_frame', 'N/A'):.4f}" if not np.isnan(frame_metrics.get('auc_frame', np.nan)) else "AUC-Frame: N/A")
        print(f"FPS: {fps_results['fps']:.2f}" if fps_results else "FPS: N/A")
        print(f"mTTA: {tta_metrics.get('mtta', 'N/A'):.4f}" if tta_metrics and not np.isnan(tta_metrics.get('mtta', np.nan)) else "mTTA: N/A")
        print(f"mRESPONSE: {response_metrics.get('mresponse', 'N/A'):.4f}" if response_metrics and not np.isnan(response_metrics.get('mresponse', np.nan)) else "mRESPONSE: N/A")
        print("========================")