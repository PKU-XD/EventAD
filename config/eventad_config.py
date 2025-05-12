import argparse
import yaml
from pathlib import Path

def parse_config(args, config: Path):
    with config.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            if k not in args:
                setattr(args, k, v)
        return args

def parse_eventad_args():
    """Parse command line arguments for EventAD model training"""
    parser = argparse.ArgumentParser(description='EventAD Model Training')
    
    # ===== Original DAGR parameters =====
    # Base path parameters
    parser.add_argument('--dataset_directory', type=Path, help="Path to the directory containing the dataset.",
                        default="/home/handsomexd/EventAD/data/detector/ROL")
    parser.add_argument('--output_directory', type=Path, default="/home/handsomexd/EventAD/log", 
                        help="Path to the logging directory.")
    parser.add_argument("--checkpoint", type=Path, default="/home/handsomexd/EventAD/checkpoints/detector/dagr_s_50.pth", 
                        help="Path to the directory containing the checkpoint.")
    parser.add_argument("--img_net", default="resnet50", type=str)
    parser.add_argument("--img_net_checkpoint", type=Path, default=argparse.SUPPRESS)

    # Configuration file parameters
    parser.add_argument("--config", type=Path, default="/home/handsomexd/EventAD/config/dagr-s-dsec.yaml")
    parser.add_argument("--use_image", default=True, action="store_true")
    parser.add_argument("--no_events", action="store_true")
    parser.add_argument("--keep_temporal_ordering", action="store_true")
    parser.add_argument("--split", default="/home/handsomexd/EventAD/config/rol_split.yaml", help="split dataset for rol")

    # Task parameters
    parser.add_argument("--task", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--dataset", default=argparse.SUPPRESS, type=str)

    # Graph parameters
    parser.add_argument('--radius', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--time_window_us', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max_neighbors', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--n_nodes', default=argparse.SUPPRESS, type=int)

    # Learning parameters
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--epochs', default=100, type=int)

    # Network parameters
    parser.add_argument("--activation", default=argparse.SUPPRESS, type=str, 
                        help="Can be one of ['Hardshrink', 'Hardsigmoid', 'Hardswish', 'ReLU', 'ReLU6', 'SoftShrink', 'HardTanh']")
    parser.add_argument("--edge_attr_dim", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--aggr", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--kernel_size", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--pooling_aggr", default=argparse.SUPPRESS, type=str)

    parser.add_argument("--base_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--after_pool_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument('--net_stem_width', default=argparse.SUPPRESS, type=float)
    parser.add_argument("--yolo_stem_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--num_scales", default=argparse.SUPPRESS, type=int)
    parser.add_argument('--pooling_dim_at_output', default=argparse.SUPPRESS)
    parser.add_argument('--weight_decay', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--clip', default=argparse.SUPPRESS, type=float)

    # Data augmentation parameters - using SUPPRESS instead of None
    parser.add_argument('--aug_p_flip', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--aug_trans', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--aug_zoom', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--l_r', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--no_eval', action="store_true")
    parser.add_argument('--tot_num_epochs', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--run_test', default=True, action="store_true")
    parser.add_argument('--num_interframe_steps', type=int, default=6)
    
    # ===== EventAD specific parameters =====
    # Model related parameters
    parser.add_argument('--x_dim', type=int, default=64, help='Feature dimension')
    parser.add_argument('--h_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--fps', type=float, default=20.0, help='Frames per second')
    
    # Training related parameters
    parser.add_argument('--experiment_name', type=str, default='eventad_dagr_experiment', help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate, stop early if below this value')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
    
    # Checkpoint related parameters
    parser.add_argument('--pretrained_model', type=str, default='', help='Pretrained model path for continued training')
    
    # Image size parameters
    parser.add_argument('--height', type=int, default=720, help='Image height')
    parser.add_argument('--width', type=int, default=1080, help='Image width')
    
    # EventAD configuration file
    parser.add_argument('--eventad_config', type=str, default='', help='EventAD configuration file path')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # If configuration file is specified, load parameters from it
    if args.config:
        args = parse_config(args, args.config)
    
    if args.eventad_config:
        args = parse_config(args, Path(args.eventad_config))
    
    # Ensure paths are Path objects
    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)
    
    if hasattr(args, 'checkpoint') and args.checkpoint:
        args.checkpoint = Path(args.checkpoint)
    
    # Set number of worker threads
    args.num_workers = 4  # Default value
    
    # Set default values for data augmentation parameters if they don't exist
    # These are required parameters, but may not exist after using SUPPRESS
    if not hasattr(args, 'aug_p_flip'):
        args.aug_p_flip = 0.5  # Default flip probability
    
    if not hasattr(args, 'aug_trans'):
        args.aug_trans = 0.1  # Default translation amount
    
    if not hasattr(args, 'aug_zoom'):
        args.aug_zoom = 1.5  # Default zoom factor
    
    return args

# If this file is run directly, print all parameters
if __name__ == "__main__":
    args = parse_eventad_args()
    print("===== EventAD Arguments =====")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")