import numpy as np
import torch
import csv
import argparse

from utils.utils import setup_directories, load_toa_values, find_best_checkpoint
from utils.fps import measure_fps
from utils.result import setup_result_file, save_metrics, create_metrics_summary
from utils.test import collect_predictions
from utils.evaluation import calculate_bbox_metrics, calculate_frame_metrics, calculate_tta_metrics,calculate_response_metrics
from utils.base import BaseEventAD
from config.eventad_config import parse_eventad_args

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class EventADTester(BaseEventAD):
    def __init__(self, args):
        """
        EventAD model tester
        
        Args:
            args: Configuration parameters
        """
        super().__init__(args)
        
        # Create necessary directories
        dirs = setup_directories(self.args.output_dir, self.args.experiment_name, mode="test")
        self.result_dir = dirs['result_dir']

        # Load TOA values
        self.video_toa = load_toa_values(self.args.toa)

        # Load test data
        self.test_dataset, self.test_loader = self.setup_data_loader('val', self.args.batch_size, shuffle=False)
        
        # Load DAGR model
        self.ema = self.setup_dagr_model(self.test_dataset)
        
        # Find best checkpoint
        checkpoint_path = find_best_checkpoint(self.args)
        
        # Initialize model
        self.model, self.checkpoint_info = self.setup_model(self.ema.ema, checkpoint_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create result recording file
        self.result_file = setup_result_file(self.result_dir, self.args, self.checkpoint_info)

    def test(self):
        """Test model and generate evaluation report"""
        print("Starting model testing...")
        
        # Collect prediction results
        results = collect_predictions(self.model, self.test_loader, self.device, self.video_toa, self.args.threshold)
        
        # Calculate sample-level metrics
        bbox_metrics = calculate_bbox_metrics(
            results['all_labels'], 
            results['all_scores'],
        )
        
        # Calculate frame-level metrics
        frame_metrics = calculate_frame_metrics(results['frame_data'])
        
        # Calculate TTA metrics
        tta_metrics = calculate_tta_metrics(
            results['video_predictions'], 
            results['video_first_anomaly'], 
            self.video_toa
        )
        
        fps_results = None
        if hasattr(self.args, 'measure_fps') and self.args.measure_fps:
            fps_results = measure_fps(
                self.model,
                self.test_loader,
                self.device,
                warmup_batches=self.args.fps_warmup_batches,
                num_batches=self.args.fps_num_batches
            )
            
            # Add FPS results to result file
            with open(self.result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['FPS (frames/second)', f"{fps_results['fps']:.2f}"])
        
        # Calculate RESPONSE metrics
        response_metrics = calculate_response_metrics(
                results['video_predictions'], 
                fps=fps_results['fps']  
            )
        # Save all metrics
        save_metrics(self.result_file, bbox_metrics, frame_metrics, tta_metrics, response_metrics)

        # Create metrics summary
        create_metrics_summary(
            self.result_dir,
            self.args,
            bbox_metrics,
            frame_metrics,
            tta_metrics,
            response_metrics,
            self.checkpoint_info,
            fps_results
        )

        print(f"Testing completed! Results saved in: {self.result_dir}")


def add_test_args(parser):
    """Add test-specific command line arguments"""
    parser.add_argument('--test_checkpoint', type=str, default='./checkpoints/detector/best_rol.pth', 
                        help='Path to specific checkpoint to test, if not specified uses best AP model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold, default is 0.5')
    parser.add_argument('--toa', type=str, default="./config/toa_values.json",
                            help='TOA values file path, default is config/toa_values.json')
    # FPS related parameters
    parser.add_argument('--measure_fps', default=True, action='store_true',
                        help='Whether to measure inference FPS')
    parser.add_argument('--fps_warmup_batches', type=int, default=70,
                        help='Number of warmup batches before FPS measurement')
    parser.add_argument('--fps_num_batches', type=int, default=20,
                        help='Number of batches used for FPS measurement')
    
    return parser


if __name__ == '__main__':
    # Get parameters using configuration parsing function
    args = parse_eventad_args()
    
    # Add test-specific parameters
    parser = argparse.ArgumentParser(description='EventAD Model Testing')
    parser = add_test_args(parser)
    test_args, _ = parser.parse_known_args()
    
    # Add test-specific parameters to args
    for arg in vars(test_args):
        setattr(args, arg, getattr(test_args, arg))
    
    # Set random seed to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create and start tester
    tester = EventADTester(args)
    tester.test()