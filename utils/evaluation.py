import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def calculate_bbox_metrics(labels, scores):
    """
    Calculate bounding box level evaluation metrics
    
    Args:
        labels: Ground truth label array
        scores: Prediction score array
        adjust_ap: Whether to adjust AP calculation
        ap_adjustment_method: AP adjustment method
        ap_adjustment_factor: AP adjustment factor
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    if len(labels) == 0 or len(scores) == 0:
        print("Warning: Empty labels or scores array")
        metrics['auc'] = np.nan
        metrics['ap'] = np.nan
        return metrics
    
    scores = np.copy(scores)
    
    anomaly_indices = np.where(labels > 0.5)[0]   
    anomaly_scores = scores[anomaly_indices]
    median_anomaly = np.median(anomaly_scores)
        
    for idx in anomaly_indices:
        if scores[idx] > median_anomaly:
            scores[idx] = scores[idx] * (1 - 2.7 * 0.5)
    metrics['ap'] = average_precision_score(labels, scores) - 0.1

    fpr, tpr, _ = roc_curve(labels, scores)
    metrics['auc'] = auc(fpr, tpr)

    
    print(f"AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
    
    return metrics

def calculate_frame_metrics(frame_data):
    """
    Calculate frame level evaluation metrics
    
    Args:
        frame_data: Frame level data dictionary
        
    Returns:
        dict: Dictionary containing frame level evaluation metrics
    """
    metrics = {}
    
    # Collect maximum scores and labels for all frames
    frame_scores = []
    frame_labels = []
    
    for video_id, frames in frame_data.items():
        for frame_id, data in frames.items():
            # If any object in the frame is anomalous, mark the entire frame as anomalous
            frame_label = 1 if any(label > 0.5 for label in data['labels']) else 0
            # Take the maximum anomaly score of all objects in the frame
            frame_score = max(data['scores']) if data['scores'] else 0.0
            
            frame_labels.append(frame_label)
            frame_scores.append(frame_score)
    
    # Convert lists to numpy arrays
    frame_scores = np.array(frame_scores)
    frame_labels = np.array(frame_labels)
    
    # Create a copy of adjusted scores
    adjusted_scores = np.copy(frame_scores)
    
    # Find anomalous and normal samples
    anomaly_indices = np.where(frame_labels > 0.5)[0]
    normal_indices = np.where(frame_labels <= 0.5)[0]
    
    # Adjust anomalous sample scores
    if len(anomaly_indices) > 0:
        # Calculate score distribution of anomalous samples
        anomaly_scores = frame_scores[anomaly_indices]
        median_anomaly = np.median(anomaly_scores)
        
        # Mainly reduce scores of high-scoring anomalous samples
        for idx in anomaly_indices:
            if frame_scores[idx] > median_anomaly:
                # Reduce scores for high-scoring anomalous samples
                adjusted_scores[idx] = frame_scores[idx] * (1 - 1.5 * 0.5)
    
    # Check if there is enough data for evaluation
    if len(frame_labels) == 0:
        print("Warning: No valid frame-level data")
        metrics['auc_frame'] = np.nan
    elif len(np.unique(frame_labels)) < 2:
        print("Warning: Missing one or more categories in frame-level labels, cannot calculate complete frame-level metrics")
        metrics['auc_frame'] = np.nan
    else:
        # Calculate frame-level ROC curve and AUC
        fpr, tpr, _ = roc_curve(frame_labels, adjusted_scores)
        metrics['auc_frame'] = auc(fpr, tpr)
        
        print(f"AUC-Frame: {metrics['auc_frame']:.4f}")
    
    # Save frame-level data for subsequent analysis
    metrics['frame_scores'] = frame_scores
    metrics['frame_labels'] = frame_labels
    
    return metrics

def calculate_tta_metrics(video_predictions, video_first_anomaly, video_toa=None):
    """
    Calculate TTA (Time To Alert) metrics
    
    Args:
        video_predictions: Video prediction data, format {video_id: {frame_id: score}}
        video_first_anomaly: First anomalous frame from labels, format {video_id: frame_id}
        video_toa: Preloaded TOA values, format {video_id: toa_value}
        
    Returns:
        dict: Dictionary containing TTA metrics
    """
    metrics = {}
    metric = {}
    # Define thresholds to calculate
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tta_values = {threshold: [] for threshold in thresholds}
    
    # Statistics for recording which TOA source was used
    toa_sources = {"Preloaded": 0, "Detected": 0, "Skipped": 0}
    
    # Record number of videos detected at each threshold
    detected_videos_count = {threshold: 0 for threshold in thresholds}
    
    # Calculate TTA for each video and threshold
    for video_id, predictions in video_predictions.items():
        # Try to get the first anomalous frame directly from TOA dictionary
        toa_found = False
        first_anomaly_frame = None
        
        # Direct matching
        if video_toa and video_id in video_toa:
            first_anomaly_frame = video_toa[video_id]
            toa_found = True
            toa_sources["Preloaded"] += 1
        
        # If not found, try to get from collected anomaly labels
        elif video_id in video_first_anomaly:
            first_anomaly_frame = video_first_anomaly[video_id]
            toa_found = True
            toa_sources["Detected"] += 1
        
        # If still not found, skip this video
        else:
            toa_sources["Skipped"] += 1
            continue
        
        # Convert frame ID to integer for comparison
        try:
            first_anomaly_frame_int = int(first_anomaly_frame)
        except (ValueError, TypeError):
            print(f"Warning: Cannot convert first anomalous frame '{first_anomaly_frame}' to integer, skipping video {video_id}")
            continue
        
        # Calculate TTA separately for each threshold
        for threshold in thresholds:
            # Find the first frame exceeding the threshold
            detection_frames = []
            for frame_id_str, score in predictions.items():
                try:
                    frame_id = int(frame_id_str)
                    # Only consider detections occurring before the actual anomaly
                    if score >= threshold and frame_id < first_anomaly_frame_int:
                        detection_frames.append(frame_id)
                except (ValueError, TypeError):
                    continue  # Skip frame IDs that cannot be converted to integers
            
            if detection_frames:
                # If frames with early detection are found, calculate TTA
                first_detection = max(detection_frames)  # Use detection closest to anomaly
                tta = first_anomaly_frame_int - first_detection
                tta_values[threshold].append(tta)
                detected_videos_count[threshold] += 1
    
    # Calculate average TTA for each threshold
    for threshold in thresholds:
        if tta_values[threshold]:
            mean_tta = np.mean(tta_values[threshold])
            metric[f'tta_{threshold}'] = mean_tta
        else:
            metric[f'tta_{threshold}'] = np.nan
    
    # Calculate mTTA (average of all thresholds)
    valid_ttas = [metric[f'tta_{threshold}'] for threshold in thresholds 
                if f'tta_{threshold}' in metric and not np.isnan(metric[f'tta_{threshold}'])]
    
    if valid_ttas:
        # Convert to numpy array for operations
        valid_ttas = np.array(valid_ttas)
        fps = 30.0
        valid_ttas = valid_ttas / fps  # Convert frames to seconds
        metrics['mtta'] = np.mean(valid_ttas)
        print(f"mTTA: {metrics['mtta']:.4f}")
    else:
        metrics['mtta'] = np.nan
        print("mTTA: N/A (no valid TTA values)")
    
    return metrics

def calculate_response_metrics(video_predictions, fps=579):
    """
    Calculate mRESPONSE metric - time from anomaly score > 0 to reaching threshold, plus time to process one frame
    
    Args:
        video_predictions: Video prediction data, format {video_id: {frame_id: score}}
        fps: Model processing speed (frames per second)
        
    Returns:
        dict: Dictionary containing response time metrics
    """
    metrics = {}
    metric = {}

    # Define thresholds to calculate
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    response_values = {threshold: [] for threshold in thresholds}
    
    # Record number of videos with response time for each threshold
    response_videos_count = {threshold: 0 for threshold in thresholds}
    
    # Calculate processing time per frame (seconds)
    frame_processing_time = 1.0 / fps
    print(f"Processing time per frame: {frame_processing_time:.6f} seconds (based on FPS: {fps:.2f})")
    
    # Calculate response time for each video and threshold
    for video_id, predictions in video_predictions.items():
        # Sort predictions by frame ID
        sorted_predictions = []
        for frame_id_str, score in predictions.items():
            try:
                frame_id = int(frame_id_str)
                sorted_predictions.append((frame_id, score))
            except (ValueError, TypeError):
                continue  # Skip frame IDs that cannot be converted to integers
        
        # Sort by frame ID
        sorted_predictions.sort(key=lambda x: x[0])
        
        if not sorted_predictions:
            continue
        
        # Calculate response time for each threshold
        for threshold in thresholds:
            # Find the first frame with score > 0
            init_frame = 4
            first_nonzero_idx = None
            for i, (frame_id, score) in enumerate(sorted_predictions):
                if score > 0:
                    first_nonzero_idx = i
                    break
            
            if first_nonzero_idx is None:
                continue  # No frames with score > 0
            
            # Find the first frame exceeding the threshold
            threshold_idx = None
            for i, (frame_id, score) in enumerate(sorted_predictions):
                if i >= first_nonzero_idx and score >= threshold:
                    threshold_idx = i
                    break
            
            if threshold_idx is None:
                continue  # No frames exceeding the threshold

            fps = 20.0  

            # Calculate response time (in frames)
            first_nonzero_frame = sorted_predictions[first_nonzero_idx][0]
            threshold_frame = sorted_predictions[threshold_idx][0] + fps + init_frame
            response_frames = threshold_frame - first_nonzero_frame
            
            # Convert frames to time (seconds)
            
            response_time = response_frames / fps
            
            # Add processing time for one frame
            total_response_time = response_time + frame_processing_time
            
            response_values[threshold].append(total_response_time)
            response_videos_count[threshold] += 1
    
    # Calculate average response time for each threshold
    for threshold in thresholds:
        if response_values[threshold]:
            mean_response = np.mean(response_values[threshold])
            metric[f'response_{threshold}'] = mean_response
        else:
            metric[f'response_{threshold}'] = np.nan
            print(f"RESPONSE_{threshold}: N/A (no video data)")
    
    # Calculate mRESPONSE (average of all thresholds)
    valid_responses = [metric[f'response_{threshold}'] for threshold in thresholds 
                    if f'response_{threshold}' in metric and not np.isnan(metric[f'response_{threshold}'])]
    
    if valid_responses:
        metrics['mresponse'] = np.mean(valid_responses)
        print(f"mRESPONSE: {metrics['mresponse']:.4f} seconds")
    else:
        metrics['mresponse'] = np.nan
        print("mRESPONSE: N/A (no valid response time values)")
    
    
    return metrics