import os
import torch
import numpy as np
from tqdm import tqdm
from dagr.utils.buffers import format_data
from utils.utils import parse_sample_id


def collect_predictions(model, test_loader, device, video_toa=None, threshold=0.5):
    """
    Collect model predictions on test set
    
    Args:
        model: Model to test
        test_loader: Test data loader
        device: Computation device
        video_toa: Video TOA values dictionary (optional)
        threshold: Threshold
        
    Returns:
        dict: Dictionary containing prediction results
    """
    print("Starting to collect prediction results...")
    
    # For storing sample-level prediction results
    all_preds = []
    all_labels = []
    all_scores = []
    sample_ids = []
    
    # Data structure for frame-level evaluation
    frame_data = {}  # Format: {video_id: {frame_id: {'scores': [], 'labels': []}}}
    
    # Data structure for TTA calculation
    video_first_anomaly = {}  # Format: {video_id: first_anomaly_frame}
    video_predictions = {}  # Format: {video_id: {frame_id: max_score}}
    
    valid_batch_count = 0
    skipped_batch_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Testing Progress")):
            try:
                # Move data to device
                data = data.to(device)
                data = format_data(data)
                
                # Get sample IDs or create default IDs
                if hasattr(data, 'sample_id'):
                    batch_ids = data.sample_id
                else:
                    batch_ids = [f"batch_{i}_sample_{j}" for j in range(len(data.x))]
                
                # Check bounding box data
                if not hasattr(data, 'bbox') or data.bbox is None or (isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] == 0):
                    print(f"Test batch {i} has no valid bounding boxes, skipping")
                    skipped_batch_count += 1
                    continue
                
                # Get labels
                labels = data.bbox[:, 4]  # 5th column is class label
                
                # Get video ID list
                if hasattr(data, 'sequence'):
                    video_ids = data.sequence
                    if isinstance(video_ids, str):
                        video_ids = [video_ids] * len(labels)  # If it's a single string, duplicate as list
                else:
                    video_ids = [f"batch_{i}"] * len(labels)  # Default video ID
            
                
                # Record true anomaly frames (for TTA calculation)
                for label_idx, (label, vid_id) in enumerate(zip(labels, video_ids)):
                    if label.item() > 0.5:  # Anomaly label
                        # Get frame ID
                        if isinstance(batch_ids, list) and label_idx < len(batch_ids):
                            _, frame_id, _ = parse_sample_id(batch_ids[label_idx])
                            
                            # Update first anomaly frame for the video
                            if vid_id not in video_first_anomaly or frame_id < video_first_anomaly[vid_id]:
                                video_first_anomaly[vid_id] = frame_id
                
                # Forward pass
                losses, batch_outputs, batch_labels = model(data, labels, testing=True)
                
                valid_batch_count += 1
                
                # Collect predictions and labels
                for j, (frame_outputs, frame_labels) in enumerate(zip(batch_outputs, batch_labels)):
                    # Get video ID for current frame
                    curr_video_id = video_ids[j] if j < len(video_ids) else f"batch_{i}"
                    
                    for k, (output, label) in enumerate(zip(frame_outputs, frame_labels)):
                        try:
                            # Extract sample ID
                            if isinstance(batch_ids, list) and j < len(batch_ids):
                                sample_id = f"{batch_ids[j]}_{k}"
                            else:
                                sample_id = f"batch_{i}_frame_{j}_obj_{k}"
                            
                            # Parse sample ID to get frame ID and object ID
                            _, frame_id, obj_id = parse_sample_id(sample_id)
                            
                            # Use video ID corresponding to current frame
                            vid_id = curr_video_id
                            
                            # Handle different output tensor shapes
                            if output.dim() == 1:  # If it's a 1D tensor [2]
                                score = output[1]  # Index 1 corresponds to anomaly class
                            elif output.dim() == 2:  # If it's a 2D tensor [1, 2]
                                score = output[0, 1]  # Use multi-dimensional indexing
                            else:
                                print(f"Warning: Unexpected output dimensions: {output.dim()}, shape: {output.shape}")
                                continue
                            
                            # Add to sample-level results
                            all_scores.append(score.item())
                            all_labels.append(label.item())
                            sample_ids.append(sample_id)
                            
                            # Add to frame-level results
                            if vid_id not in frame_data:
                                frame_data[vid_id] = {}
                            
                            if frame_id not in frame_data[vid_id]:
                                frame_data[vid_id][frame_id] = {'scores': [], 'labels': []}
                            
                            frame_data[vid_id][frame_id]['scores'].append(score.item())
                            frame_data[vid_id][frame_id]['labels'].append(label.item())
                            
                            # Add to TTA calculation data
                            if vid_id not in video_predictions:
                                video_predictions[vid_id] = {}
                            
                            if frame_id not in video_predictions[vid_id]:
                                video_predictions[vid_id][frame_id] = 0.0
                            
                            # Update maximum anomaly score for the frame
                            video_predictions[vid_id][frame_id] = max(
                                video_predictions[vid_id][frame_id], 
                                score.item()
                            )
                            
                        except Exception as e:
                            print(f"Error processing output: {e}, output shape: {output.shape if hasattr(output, 'shape') else 'unknown'}")
                            continue
            
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                skipped_batch_count += 1
                continue
    
    # If no valid data, raise error
    if not all_labels:
        raise RuntimeError("No valid predictions or labels collected, please check dataset and model")
    
    print(f"Testing complete: Valid batches {valid_batch_count}/{len(test_loader)}, Skipped batches {skipped_batch_count}")
    print(f"Collected predictions for {len(all_labels)} samples")
    
    # Calculate predicted labels using threshold
    all_preds = [1 if score > threshold else 0 for score in all_scores]
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    return {
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_scores': all_scores,
        'sample_ids': sample_ids,
        'frame_data': frame_data,
        'video_first_anomaly': video_first_anomaly,
        'video_predictions': video_predictions,
        'valid_batch_count': valid_batch_count,
        'skipped_batch_count': skipped_batch_count
    }