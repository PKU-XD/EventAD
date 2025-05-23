import argparse
import yaml

from pathlib import Path


def BASE_FLAGS():
    parser = argparse.ArgumentParser("")
    # "/home/datasets/XDD/DSEC"
    # "/home/handsomexd/EventAD/data/detector/ROL"
    # "/home/handsomexd/EventAD/checkpoints/detector/dagr_s_50.pth"
    # "/home/handsomexd/EventAD/log/ROL/detection/vocal-elevator-22/model_47.pth" 
    # "/home/handsomexd/EventAD/log/ROL/detection/morning-totem-25/model_31.pth" resume
    parser.add_argument('--dataset_directory', type=Path, help="Path to the directory containing the dataset.",default="/home/handsomexd/EventAD/data/detector/ROL")
    parser.add_argument('--output_directory', type=Path, default="/home/handsomexd/EventAD/log", help="Path to the logging directory.")
    parser.add_argument("--checkpoint", type=Path, default="/home/handsomexd/EventAD/checkpoints/detector/dagr_s_50.pth", help="Path to the directory containing the checkpoint.")
    parser.add_argument("--img_net", default="resnet50", type=str)
    parser.add_argument("--img_net_checkpoint", type=Path, default=argparse.SUPPRESS)

    parser.add_argument("--config", type=Path, default="/home/handsomexd/EventAD/config/dagr-s-dsec.yaml")
    parser.add_argument("--use_image", default=True,action="store_true")
    parser.add_argument("--no_events", action="store_true")
    parser.add_argument("--keep_temporal_ordering", action="store_true")
    parser.add_argument("--split", default="/home/handsomexd/EventAD/config/rol_split.yaml",help="split dataset for rol")

    # task params
    parser.add_argument("--task", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--dataset", default=argparse.SUPPRESS, type=str)

    # graph params
    parser.add_argument('--radius', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--time_window_us', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max_neighbors', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--n_nodes', default=argparse.SUPPRESS, type=int)

    # learning params
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--optimizer',default='adam',type=str)
    parser.add_argument('--lr',default=0.003,type=float)
    parser.add_argument('--lr_scheduler',default='cosine',type=str)
    parser.add_argument('--epochs',default=100,type=int)

    # network params
    parser.add_argument("--activation", default=argparse.SUPPRESS, type=str, help="Can be one of ['Hardshrink', 'Hardsigmoid', 'Hardswish', 'ReLU', 'ReLU6', 'SoftShrink', 'HardTanh']")
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

    parser.add_argument('--aug_p_flip', default=argparse.SUPPRESS, type=float)

    return parser

def FLAGS():
    parser = BASE_FLAGS()

    # learning params
    parser.add_argument('--aug_trans', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--aug_zoom', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--l_r', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--no_eval', action="store_true")
    parser.add_argument('--tot_num_epochs', default=argparse.SUPPRESS, type=int)

    parser.add_argument('--run_test', default = True,action="store_true")

    parser.add_argument('--num_interframe_steps', type=int, default=6)

    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)

    if "checkpoint" in args:
        args.checkpoint = Path(args.checkpoint)

    return args

def FLOPS_FLAGS():
    parser = BASE_FLAGS()

    # for flop eval
    parser.add_argument("--check_consistency", action="store_true")
    parser.add_argument("--dense", action="store_true")

    # for runtime eval
    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)

    if "checkpoint" in args:
        args.checkpoint = Path(args.checkpoint)

    return args


def parse_config(args: argparse.ArgumentParser, config: Path):
    with config.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            if k not in args:
                setattr(args, k, v)
        return args
