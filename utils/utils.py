import os
import time
import re
import json
import torch
import numpy as np
from pathlib import Path

def setup_directories(base_dir, experiment_name, mode="train"):
    """
    Create directory structure needed for experiments
    
    Args:
        base_dir: Base output directory
        experiment_name: Experiment name
        mode: 'train' or 'test' mode
        
    Returns:
        dict: Dictionary containing various directory paths
    """
    # Create unique experiment directory using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if mode == "train":
        # Directory structure for training mode
        result_dir = os.path.join(base_dir, "results", f"{experiment_name}_{timestamp}")
        model_dir = os.path.join(base_dir, "models", f"{experiment_name}_{timestamp}")
        
        # Create directories
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        return {
            'result_dir': result_dir,
            'model_dir': model_dir,
            'timestamp': timestamp
        }
    else:
        # Directory structure for testing mode
        result_dir = os.path.join(base_dir, "test_results", f"{experiment_name}_{timestamp}")
        
        # Create directories
        os.makedirs(result_dir, exist_ok=True)
        
        return {
            'result_dir': result_dir,
            'timestamp': timestamp
        }

def load_toa_values(toa_file):
    """
    Load TOA (Time of Alert) values for videos
    
    Args:
        toa_file: Path to TOA values file
        
    Returns:
        dict: Mapping from video ID to TOA value
    """
    if not os.path.exists(toa_file):
        print(f"Warning: TOA values file {toa_file} does not exist, will use detected anomaly frames")
        return {}
    
    with open(toa_file, 'r') as f:
        toa_dict = json.load(f)
    
    print(f"Loaded TOA values for {len(toa_dict)} videos")
    return toa_dict

def parse_sample_id(sample_id):
    """
    Parse sample ID to extract frame ID and object ID
    
    Args:
        sample_id: Sample ID string
        
    Returns:
        tuple: (video_id, frame_id, object_id) - video_id is always None
    """
    # Try to extract information from sample ID
    if isinstance(sample_id, str):
        # Extract frame ID and object ID
        frame_match = re.search(r'frame_(\d+)', sample_id)
        obj_match = re.search(r'obj_(\d+)', sample_id)
        
        frame_id = int(frame_match.group(1)) if frame_match else 0
        obj_id = int(obj_match.group(1)) if obj_match else 0
        
        # Video ID will be obtained from sequence attribute of data object
        return None, frame_id, obj_id
    else:
        # If not a string, return default values
        return None, 0, 0

def find_best_checkpoint(args):
    """
    Find best checkpoint path
    
    Args:
        args: Configuration parameters
        
    Returns:
        str: Checkpoint path
    """
    # Prioritize checkpoint specified in command line arguments
    if hasattr(args, 'test_checkpoint') and args.test_checkpoint:
        return args.test_checkpoint
    
    # Otherwise try to load best AP model
    model_dir = Path(args.output_dir) / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    # Find latest experiment directory
    experiment_dirs = sorted(list(model_dir.glob(f"{args.experiment_name}_*")), reverse=True)
    if not experiment_dirs:
        raise FileNotFoundError(f"No directories matching experiment name: {args.experiment_name}")
    
    latest_exp_dir = experiment_dirs[0]
    
    # Prioritize best AP model
    best_ap_path = latest_exp_dir / "best_ap_model.pth"
    best_auc_path = latest_exp_dir / "best_auc_model.pth"
    latest_path = latest_exp_dir / "latest_checkpoint.pth"
    
    if best_ap_path.exists():
        return best_ap_path
    elif best_auc_path.exists():
        return best_auc_path
    elif latest_path.exists():
        return latest_path
    else:
        raise FileNotFoundError(f"Could not find any checkpoints in directory {latest_exp_dir}")