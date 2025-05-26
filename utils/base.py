import os
import torch
from torch_geometric.data import DataLoader
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from models.EventAD import EventADModel
from dagr.utils.buffers import format_data

class BaseEventAD:
    """Base class for EventAD models with shared functionality"""
    
    def __init__(self, args):
        """
        Initialize the base class
        
        Args:
            args: configuration parameters
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_dagr_model(self, dataset):
        """
        Load DAGR model
        
        Args:
            dataset: dataset object, used to get height and width
            
        Returns:
            ModelEMA: EMA object wrapping the DAGR model
        """
        print("Loading DAGR model...")
        
        dagr_model = DAGR(
            self.args, 
            height=dataset.height, 
            width=dataset.width
        ).to(self.device)
        
        ema = ModelEMA(dagr_model)
        
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        ema.ema.load_state_dict(checkpoint['ema'])
        
        ema.ema.cache_luts(
            radius=self.args.radius, 
            height=dataset.height, 
            width=dataset.width
        )
        
        print("DAGR model loaded")
        return ema
    
    def setup_model(self, dagr_model, checkpoint_path=None):
        """
        Initialize EventAD model
        
        Args:
            dagr_model: DAGR model object
            checkpoint_path: optional checkpoint path
            
        Returns:
            tuple: (model, checkpoint info)
        """
        model = EventADModel(
            dagr_model=dagr_model,
            x_dim=self.args.x_dim,
            h_dim=self.args.h_dim,
            n_frames=self.args.n_frames,
            fps=self.args.fps
        ).to(self.device)
        
        checkpoint_info = {
            'path': 'N/A',
            'epoch': 0,
            'best_auc': 0.0,
            'best_ap': 0.0
        }
        
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            checkpoint_info = {
                'path': str(checkpoint_path),
                'epoch': checkpoint.get('epoch', 0),
                'best_auc': checkpoint.get('best_auc', 0.0),
                'best_ap': checkpoint.get('best_ap', 0.0)
            }
        
        print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, checkpoint_info
    
    def setup_data_loader(self, split, batch_size, shuffle=True):
        """
        Set up data loader
        
        Args:
            split: dataset split, 'train' or 'val'
            batch_size: batch size
            shuffle: whether to shuffle data
            
        Returns:
            tuple: (dataset, data loader)
        """
        print(f"Loading {split} dataset...")
        
        transform = Augmentations.transform_training if split == 'train' else Augmentations.transform_testing
        
        dataset = DSEC(
            self.args,
            self.args.dataset_directory, 
            split,
            transform,
            debug=False, 
            min_bbox_diag=15, 
            min_bbox_height=10
        )
        
        loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            follow_batch=['bbox', 'bbox0'],
            num_workers=self.args.num_workers
        )
        
        print(f"{split} data: {len(dataset)} samples")
        return dataset, loader
