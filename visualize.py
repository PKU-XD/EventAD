import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import json

from models.EventAD import EventADModel
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from config.eventad_config import parse_eventad_args
from dagr.utils.buffers import format_data

class EventADVisualizer:
    def __init__(self, args):
        """
        EventAD model visualization tool
        
        Args:
            args: Configuration parameters
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create necessary directories
        self.setup_directories()
        
        # Load test data
        self.setup_data_loader()
        
        # Load DAGR model
        self.setup_dagr_model()
        
        # Initialize model
        self.setup_model()
        
        # Set colormap for visualization
        self.setup_colormap()
        
    def setup_directories(self):
        """Create necessary directories"""
        # Create a unique visualization directory with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.vis_dir = os.path.join(self.args.output_dir, "visualizations", f"{self.args.experiment_name}_{timestamp}")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Create subdirectories for different visualization types
        self.event_vis_dir = os.path.join(self.vis_dir, "event_frames")
        self.bbox_vis_dir = os.path.join(self.vis_dir, "bbox_overlay")
        self.combined_vis_dir = os.path.join(self.vis_dir, "combined")
        
        os.makedirs(self.event_vis_dir, exist_ok=True)
        os.makedirs(self.bbox_vis_dir, exist_ok=True)
        os.makedirs(self.combined_vis_dir, exist_ok=True)
        
    def setup_data_loader(self):
        """Set up test data loader"""
        print("Loading test dataset...")
        
        # Load test dataset
        self.test_dataset = DSEC(
            self.args,
            self.args.dataset_directory, 
            "test",  # Use test set
            Augmentations.transform_testing,  # Use testing transform
            debug=False, 
            min_bbox_diag=15, 
            min_bbox_height=10
        )
        
        # Create test data loader with batch size 1 for visualization
        from torch_geometric.data import DataLoader
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=1,  # Use batch size 1 for visualization
            shuffle=False, 
            follow_batch=['bbox', 'bbox0'],
            num_workers=self.args.num_workers
        )
        
        print(f"Test data: {len(self.test_dataset)} samples")
        
        # Get dataset dimensions
        self.height = self.test_dataset.height
        self.width = self.test_dataset.width
        print(f"Dataset dimensions: {self.width}x{self.height}")

    def setup_dagr_model(self):
        """Load DAGR model"""
        print("Loading DAGR model...")
        
        # Create DAGR model
        self.dagr_model = DAGR(
            self.args, 
            height=self.height, 
            width=self.width
        ).to(self.device)
        
        # Create EMA model instance
        self.ema = ModelEMA(self.dagr_model)
        
        # Load pretrained weights
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        self.ema.ema.load_state_dict(checkpoint['ema'])
        
        # Cache lookup tables
        self.ema.ema.cache_luts(
            radius=self.args.radius, 
            height=self.height, 
            width=self.width
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
        
        # Determine which checkpoint to load
        if hasattr(self.args, 'vis_checkpoint') and self.args.vis_checkpoint:
            checkpoint_path = self.args.vis_checkpoint
        else:
            # Try to load the best AP model
            model_dir = Path(self.args.output_dir) / "models"
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
            
            # Find the latest experiment directory
            experiment_dirs = sorted(list(model_dir.glob(f"{self.args.experiment_name}_*")), reverse=True)
            if not experiment_dirs:
                raise FileNotFoundError(f"No directory matching experiment name: {self.args.experiment_name}")
            
            latest_exp_dir = experiment_dirs[0]
            
            # Prefer best AP model
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
                raise FileNotFoundError(f"No checkpoint found in directory {latest_exp_dir}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Print model info
        print(f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_colormap(self):
        """Set up colormap for visualization"""
        # Create colormap for event visualization
        self.colormap = plt.cm.jet
        
        # Create colormap for anomaly scores
        self.anomaly_cmap = plt.cm.RdYlGn_r  # Red for high anomaly, green for low anomaly
        
    def visualize_events(self, events, frame_idx):
        """
        Visualize events as a frame
        
        Args:
            events: Event data tensor
            frame_idx: Frame index for filename
            
        Returns:
            event_frame: Rendered event frame as numpy array
        """
        # Create an empty frame
        event_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if events is not None and len(events) > 0:
            # Extract event coordinates and polarities
            x = events[:, 0].long()
            y = events[:, 1].long()
            p = events[:, 2]  # Polarity
            t = events[:, 3]  # Timestamp
            
            # Normalize timestamps to [0, 1]
            if len(t) > 0:
                t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
                
                # Convert to numpy for visualization
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                p_np = p.cpu().numpy()
                t_np = t_norm.cpu().numpy()
                
                # Filter valid coordinates
                valid_idx = (x_np >= 0) & (x_np < self.width) & (y_np >= 0) & (y_np < self.height)
                x_np = x_np[valid_idx]
                y_np = y_np[valid_idx]
                p_np = p_np[valid_idx]
                t_np = t_np[valid_idx]
                
                # Visualize events with color based on timestamp and polarity
                for i in range(len(x_np)):
                    color = self.colormap(t_np[i])[:3]  # Get RGB from colormap
                    # Adjust brightness based on polarity
                    if p_np[i] > 0:
                        color = tuple(int(c * 255) for c in color)  # Positive events
                    else:
                        color = tuple(int(c * 128) for c in color)  # Negative events
                    
                    # Set pixel color
                    event_frame[y_np[i], x_np[i]] = color
        
        # Save event frame
        event_frame_path = os.path.join(self.event_vis_dir, f"event_frame_{frame_idx:04d}.png")
        cv2.imwrite(event_frame_path, cv2.cvtColor(event_frame, cv2.COLOR_RGB2BGR))
        
        return event_frame
    
    def visualize_bboxes(self, event_frame, bboxes, scores, frame_idx):
        """
        Visualize bounding boxes and anomaly scores on the event frame
        
        Args:
            event_frame: Event frame as numpy array
            bboxes: Bounding boxes [x1, y1, x2, y2, class_id]
            scores: Anomaly scores for each bounding box
            frame_idx: Frame index for filename
            
        Returns:
            bbox_frame: Event frame with bounding boxes overlay
        """
        # Create a copy of the event frame
        bbox_frame = event_frame.copy()
        
        # Create a figure for matplotlib
        fig, ax = plt.subplots(1, figsize=(self.width/100, self.height/100), dpi=100)
        ax.imshow(bbox_frame)
        
        # Draw bounding boxes
        if bboxes is not None and len(bboxes) > 0:
            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                x1, y1, x2, y2 = bbox[:4].int().cpu().numpy()
                class_id = int(bbox[4].item())
                
                # Determine color based on anomaly score (red for high anomaly, green for low)
                color = self.anomaly_cmap(score.item())
                
                # Create rectangle with transparency based on score
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                 edgecolor=color, facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                
                # Add text with class and score
                label = f"Class: {class_id}, Score: {score.item():.2f}"
                ax.text(x1, y1-5, label, color='white', fontsize=8, 
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
        
        # Remove axes
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Save figure
        bbox_frame_path = os.path.join(self.bbox_vis_dir, f"bbox_overlay_{frame_idx:04d}.png")
        plt.savefig(bbox_frame_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Load the saved image for returning
        bbox_frame = cv2.imread(bbox_frame_path)
        bbox_frame = cv2.cvtColor(bbox_frame, cv2.COLOR_BGR2RGB)
        
        return bbox_frame
    
    def create_combined_visualization(self, event_frame, bbox_frame, frame_idx):
        """
        Create a combined visualization with both event frame and bbox overlay
        
        Args:
            event_frame: Event frame as numpy array
            bbox_frame: Event frame with bounding boxes overlay
            frame_idx: Frame index for filename
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot event frame
        ax1.imshow(event_frame)
        ax1.set_title("Event Frame")
        ax1.axis('off')
        
        # Plot bbox overlay
        ax2.imshow(bbox_frame)
        ax2.set_title("Anomaly Detection Results")
        ax2.axis('off')
        
        # Add colorbar for anomaly scores
        sm = plt.cm.ScalarMappable(cmap=self.anomaly_cmap)
        sm.set_array([0, 1])
        cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.01)
        cbar.set_label('Anomaly Score')
        
        # Save combined visualization
        plt.tight_layout()
        combined_path = os.path.join(self.combined_vis_dir, f"combined_{frame_idx:04d}.png")
        plt.savefig(combined_path)
        plt.close(fig)
    
    def save_metadata(self, frame_idx, bboxes, scores):
        """
        Save metadata for each frame as JSON
        
        Args:
            frame_idx: Frame index
            bboxes: Bounding boxes
            scores: Anomaly scores
        """
        metadata = {
            "frame_idx": int(frame_idx),
            "bboxes": [],
            "anomaly_scores": []
        }
        
        if bboxes is not None and len(bboxes) > 0:
            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                bbox_data = bbox.cpu().numpy().tolist()
                score_data = float(score.item())
                
                metadata["bboxes"].append(bbox_data)
                metadata["anomaly_scores"].append(score_data)
        
        # Save metadata as JSON
        metadata_path = os.path.join(self.vis_dir, f"metadata_{frame_idx:04d}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def visualize(self):
        """Run visualization on test data"""
        print("Starting visualization...")
        
        # Create a summary file for all frames
        summary_path = os.path.join(self.vis_dir, "visualization_summary.csv")
        with open(summary_path, 'w') as f:
            f.write("frame_idx,num_bboxes,avg_anomaly_score,max_anomaly_score\n")
        
        # Process each batch
        frame_idx = 0
        
        for data in tqdm(self.test_loader, desc="Visualizing"):
            try:
                # Move data to device
                data = data.to(self.device)
                data = format_data(data)
                
                # Get events
                events = data.x
                
                # Get ground truth bounding boxes if available
                if hasattr(data, 'bbox') and data.bbox is not None and isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] > 0:
                    gt_bboxes = data.bbox
                    labels = data.bbox[:, 4]  # Assume 5th column is class label
                else:
                    # If no bounding boxes, create dummy labels
                    gt_bboxes = None
                    labels = torch.zeros(1, device=self.device)
                
                # Forward pass
                with torch.no_grad():
                    losses, batch_outputs, batch_labels = self.model(data, labels, testing=True)
                
                # Process each frame in the batch
                for frame_events, frame_outputs, frame_labels in zip([events], batch_outputs, batch_labels):
                    # Visualize events
                    event_frame = self.visualize_events(frame_events, frame_idx)
                    
                    # Extract bounding boxes and scores
                    if len(frame_outputs) > 0:
                        # Get predicted scores
                        scores = torch.tensor([output[1] for output in frame_outputs])  # Assume index 1 is anomaly class
                        
                        # Use ground truth bounding boxes with predicted scores
                        bboxes = gt_bboxes if gt_bboxes is not None else None
                    else:
                        scores = torch.tensor([])
                        bboxes = None
                    
                    # Visualize bounding boxes
                    bbox_frame = self.visualize_bboxes(event_frame, bboxes, scores, frame_idx)
                    
                    # Create combined visualization
                    self.create_combined_visualization(event_frame, bbox_frame, frame_idx)
                    
                    # Save metadata
                    self.save_metadata(frame_idx, bboxes, scores)
                    
                    # Update summary file
                    with open(summary_path, 'a') as f:
                        num_bboxes = len(scores) if scores is not None else 0
                        avg_score = scores.mean().item() if num_bboxes > 0 else 0
                        max_score = scores.max().item() if num_bboxes > 0 else 0
                        f.write(f"{frame_idx},{num_bboxes},{avg_score:.4f},{max_score:.4f}\n")
                    
                    frame_idx += 1
            
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Visualization completed! {frame_idx} frames processed.")
        print(f"Results saved to: {self.vis_dir}")
        
        # Create a video from the combined visualizations
        self.create_video()
    
    def create_video(self):
        """Create a video from the combined visualizations"""
        try:
            import cv2
            
            # Get all combined visualization files
            combined_files = sorted(os.listdir(self.combined_vis_dir))
            if not combined_files:
                print("No visualization files found to create video")
                return
            
            # Read the first image to get dimensions
            first_img = cv2.imread(os.path.join(self.combined_vis_dir, combined_files[0]))
            height, width, layers = first_img.shape
            
            # Create video writer
            video_path = os.path.join(self.vis_dir, "visualization_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
            
            # Add each frame to the video
            for file in tqdm(combined_files, desc="Creating video"):
                if file.endswith(".png"):
                    img_path = os.path.join(self.combined_vis_dir, file)
                    frame = cv2.imread(img_path)
                    video.write(frame)
            
            # Release video writer
            video.release()
            print(f"Video created at: {video_path}")
            
        except Exception as e:
            print(f"Error creating video: {e}")
            import traceback
            traceback.print_exc()


def add_visualization_args(parser):
    """Add visualization-specific command-line arguments"""
    parser.add_argument('--vis_checkpoint', type=str, default='', 
                        help='Specific checkpoint path to use for visualization')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Maximum number of frames to visualize (0 for all)')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5,
                        help='Threshold for highlighting high anomaly scores')
    parser.add_argument('--create_video', action='store_true',
                        help='Create a video from the visualizations')
    return parser


if __name__ == '__main__':
    # Use configuration parsing function to get parameters
    args = parse_eventad_args()
    
    # Add visualization-specific arguments
    parser = argparse.ArgumentParser(description='EventAD Visualization')
    parser = add_visualization_args(parser)
    vis_args, _ = parser.parse_known_args()
    
    # Add visualization-specific parameters to args
    for arg in vars(vis_args):
        setattr(args, arg, getattr(vis_args, arg))
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create and run visualizer
    visualizer = EventADVisualizer(args)
    visualizer.visualize()