from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Fuse_GRUNet(nn.Module):
    def __init__(self, event_dim, coord_dim):
        super(Fuse_GRUNet, self).__init__()
        
        # Projection layers
        self.event_proj = nn.Linear(event_dim, 256)
        self.coord_proj = nn.Linear(coord_dim, 256)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, 4)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Output 2 classes: normal and anomaly
        )
    
    def forward(self, event_feature, coord_feature):
        """
        Fuse event features and coordinate features
        
        Args:
            event_feature: Event feature [batch_size, event_dim]
            coord_feature: Coordinate feature [batch_size, coord_dim]
            
        Returns:
            output: Fused feature [batch_size, 2]
        """
        # Projection
        event_proj = self.event_proj(event_feature)  # [batch_size, 256]
        coord_proj = self.coord_proj(coord_feature)  # [batch_size, 256]

        
        # Ensure inputs are trainable
        if not event_proj.requires_grad:
            event_proj = event_proj.detach().requires_grad_(True)
        
        if not coord_proj.requires_grad:
            coord_proj = coord_proj.detach().requires_grad_(True)
        
        # Concatenate features
        combined = torch.cat([event_proj, coord_proj], dim=1)  # [batch_size, 256]
        
        # Fusion
        output = self.fusion(combined)  # [batch_size, 2]

        return output
    

class Cor_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Cor_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h
    

class Event_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Event_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class SpatialAttention(nn.Module):
    """
    Apply soft attention mechanism to hidden representations of all objects in a frame.
    """
    def __init__(self, h_dim):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_dim, 1))  # Hidden representation dimension
        self.softmax = nn.Softmax(dim=1)
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, h_all_in):
        """
        Args:
            h_all_in: Dictionary containing object tracking IDs and hidden representations
            
        Returns:
            Dictionary containing updated hidden representations
        """
        k = []
        v = []
        for key in h_all_in:
            v.append(h_all_in[key])
            k.append(key)

        if len(v) != 0:
            h_in = torch.cat([element for element in v], dim=1)
            m = torch.tanh(h_in)
            alpha = torch.softmax(torch.matmul(m, self.weight), 1)
            roh = torch.mul(h_in, alpha)
            list_roh = []
            for i in range(roh.size(1)):
                list_roh.append(roh[:, i, :].unsqueeze(1).contiguous())

            h_all_in = {}
            for ke, value in zip(k, list_roh):
                h_all_in[ke] = value

        return h_all_in
    

class EventADModel(nn.Module):
    def __init__(self, dagr_model, x_dim=64, h_dim=256, n_frames=2, fps=30):
        super(EventADModel, self).__init__()
        
        # DAGR model
        self.dagr_model = dagr_model
        
        # Freeze DAGR model parameters
        for param in self.dagr_model.parameters():
            param.requires_grad = False
            
        # But we need to ensure our custom layers are trainable
        self.gru_net_event = Event_GRUNet(x_dim, h_dim, n_layers=2)  # Add n_layers parameter
        self.gru_net_cor = Cor_GRUNet(4, 32, n_layers=1)  # Add n_layers parameter, coordinate feature dimension is 4 (x, y, w, h)
        self.fusion_module = Fuse_GRUNet(h_dim, 32)
        
        # Ensure these layers' parameters are trainable
        for module in [self.gru_net_event, self.gru_net_cor, self.fusion_module]:
            for param in module.parameters():
                param.requires_grad = True
                
        # Modify attention mechanism to ensure it can handle correct dimensions
        self.soft_attention = SpatialAttention(h_dim)  # Hidden dimension for event features
        self.soft_attention_cor = SpatialAttention(32)  # Hidden dimension for coordinate features
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Cache
        self.batch_feature_cache = None
        self.batch_input_hash = None
        
        # Configuration
        self.n_frames = n_frames
        self.fps = fps
        self.max_boxes = 30  # Maximum number of bounding boxes to process per frame
        
    def forward(self, data, labels=None, testing=False):
        """
        Forward pass
        
        Args:
            data: Input data
            labels: Labels
            testing: Whether in testing mode
            
        Returns:
            losses: Loss dictionary
            outputs: Output list
            output_labels: Output label list
        """
        batch_size = data.num_graphs
        device = data.x.device
        
        # Initialize loss dictionary
        losses = {'cross_entropy': 0}
        
        # Extract features
        batch_features = self.extract_features(data)
        
        # Maintain hidden states for each track_id
        h_all_in_event = {}
        h_all_out_event = {}
        h_all_in_coord = {}
        h_all_out_coord = {}
        
        all_outputs = []
        all_labels = []
        
        # Process each batch (equivalent to each frame in model.py)
        for b in range(batch_size):
            frame_outputs = []
            frame_labels = []
            
            # Get features for current batch
            prev_features = batch_features[b, 0]  # Previous frame features [max_boxes+1, feature_dim]
            curr_features = batch_features[b, 1]  # Current frame features [max_boxes+1, feature_dim]
            
            # Reset output dictionaries for each frame
            h_all_out_event = {}
            h_all_out_coord = {}
            
            # Process each bounding box
            for track_id in range(1, self.max_boxes + 1):
                # Get bounding box features for current frame
                curr_feature = curr_features[track_id]
                
                # Check if feature is valid (non-zero)
                if torch.sum(curr_feature) == 0:
                    continue
                    
                # Stringify track_id, consistent with model.py
                track_id_str = str(track_id)
                
                # Get corresponding bounding box coordinates
                if hasattr(data, 'bbox') and data.bbox is not None:
                    bbox_indices = torch.where((data.bbox_batch == b) & (data.bbox[:, 5] == track_id))[0]
                    if len(bbox_indices) > 0:
                        bbox_idx = bbox_indices[0]
                        bbox_data = data.bbox[bbox_idx]
                        
                        # Extract coordinates [x, y, w, h]
                        # Normalize coordinates (assuming width and height are data.width[b] and data.height[b])
                        width_val = data.width[b].item() if isinstance(data.width[b], torch.Tensor) else data.width[b]
                        height_val = data.height[b].item() if isinstance(data.height[b], torch.Tensor) else data.height[b]
                        
                        x, y, w, h = bbox_data[0].item(), bbox_data[1].item(), bbox_data[2].item(), bbox_data[3].item()
                        norm_coord = torch.tensor([[x/width_val, y/height_val, w/width_val, h/height_val]], 
                                            device=device).unsqueeze(0)  # [1, 1, 4]
                        
                        # Get label
                        if labels is not None:
                            target = labels[bbox_idx].long().unsqueeze(0)
                        else:
                            target = torch.tensor([0], device=device).long()
                    else:
                        continue
                else:
                    continue
                
                # Convert current feature to sequence format [1, 1, feature_dim]
                event_feature = curr_feature.unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
                
                # Check if this track_id exists in previous frame
                if track_id_str in h_all_in_event:
                    # Use hidden state from previous frame
                    h_in_event = h_all_in_event[track_id_str]
                    h_in_coord = h_all_in_coord[track_id_str]
                    
                    # Process event feature sequence
                    event_output, h_out_event = self.gru_net_event(event_feature, h_in_event)
                    
                    # Process coordinate sequence
                    coord_output, h_out_coord = self.gru_net_cor(norm_coord, h_in_coord)
                    
                    # Fuse features
                    output = self.fusion_module(event_output[:, -1, :], coord_output[:, -1, :])
                    
                    # Calculate loss
                    loss = self.ce_loss(output, target)
                    losses['cross_entropy'] += loss
                    
                    # Store output and label
                    frame_outputs.append(output)
                    frame_labels.append(target)
                    
                    # Update hidden states
                    h_all_out_event[track_id_str] = h_out_event
                    h_all_out_coord[track_id_str] = h_out_coord
                    
                else:
                    # Initialize new hidden states
                    h_in_event = torch.zeros(self.gru_net_event.n_layers, 1, 
                                        self.gru_net_event.hidden_dim, device=device)
                    h_in_coord = torch.zeros(self.gru_net_cor.n_layers, 1, 
                                        self.gru_net_cor.hidden_dim, device=device)
                    
                    # Process event feature sequence
                    event_output, h_out_event = self.gru_net_event(event_feature, h_in_event)
                    
                    # Process coordinate sequence
                    coord_output, h_out_coord = self.gru_net_cor(norm_coord, h_in_coord)
                    
                    # Fuse features
                    output = self.fusion_module(event_output[:, -1, :], coord_output[:, -1, :])
                    
                    # Calculate loss
                    loss = self.ce_loss(output, target)
                    losses['cross_entropy'] += loss
                    
                    # Store output and label
                    frame_outputs.append(output)
                    frame_labels.append(target)
                    
                    # Update hidden states
                    h_all_out_event[track_id_str] = h_out_event
                    h_all_out_coord[track_id_str] = h_out_coord
            
            # Apply attention mechanism to update hidden states
            if len(h_all_out_event) > 0:
                h_all_in_event.update(self.soft_attention(h_all_out_event))
            if len(h_all_out_coord) > 0:
                h_all_in_coord.update(self.soft_attention_cor(h_all_out_coord))
                
            if len(frame_outputs) > 0:
                all_outputs.append(frame_outputs)
                all_labels.append(frame_labels)
        
        # If no valid predictions, use zero loss
        if losses['cross_entropy'] == 0:
            losses['cross_entropy'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        return losses, all_outputs, all_labels
    
    def extract_features(self, data):
        """
        Extract features from data batch
        
        Args:
            data: DataBatch containing graph data, bounding boxes, etc.
            
        Returns:
            batch_features: Batch features [batch_size, n_frames, max_boxes+1, feature_dim]
        """
        batch_size = data.num_graphs
        device = data.x.device
        
        # Calculate hash of input for caching
        x_sum = data.x.sum().item() if hasattr(data, 'x') and data.x is not None else 0
        input_hash = hash(str(x_sum))
        
        # If cache is valid, return directly
        if self.batch_feature_cache is not None and self.batch_input_hash == input_hash:
            return self.batch_feature_cache
        
        # Use DAGR model to extract features
        with torch.no_grad():
            self.dagr_model.eval()
            # Extract features
            dagr_output = self.dagr_model.extract_features(data)
            # Get node features (from out4)
            out4 = dagr_output[1]  # DataBatch(x=[169, 64], ...)
            
            # Get node features, positions, and batch information
            node_features = out4.x  # [num_nodes, feature_dim]
            node_positions = out4.pos[:, :2] if out4.pos is not None else torch.zeros((out4.x.size(0), 2), device=device)
            node_batch = out4.batch  # [num_nodes]
            
            # Calculate global features for each batch
            global_features = torch.zeros((batch_size, node_features.size(1)), device=device)
            if node_batch is not None and node_features.size(0) > 0:
                for b in range(batch_size):
                    batch_mask = (node_batch == b)
                    if batch_mask.any():
                        global_features[b] = node_features[batch_mask].mean(dim=0)
        
        # Initialize batch features [batch_size, n_frames, max_boxes+1, feature_dim]
        feature_dim = node_features.shape[1]  # Should be 64
        batch_features = torch.zeros((batch_size, 2, self.max_boxes + 1, feature_dim), device=device)
        
        # Process bounding boxes for each batch
        for b in range(batch_size):
            # Process current frame bounding boxes
            if hasattr(data, 'bbox') and data.bbox is not None and hasattr(data, 'bbox_batch'):
                bbox_indices = torch.where(data.bbox_batch == b)[0]
                for idx in bbox_indices:
                    bbox = data.bbox[idx]
                    self._process_bbox(bbox, b, 1, node_features, node_positions, node_batch, 
                                    global_features[b], batch_features, data.width[b], data.height[b])
            
            # Process previous frame bounding boxes
            if hasattr(data, 'bbox0') and data.bbox0 is not None and hasattr(data, 'bbox0_batch'):
                bbox0_indices = torch.where(data.bbox0_batch == b)[0]
                for idx in bbox0_indices:
                    bbox0 = data.bbox0[idx]
                    self._process_bbox(bbox0, b, 0, node_features, node_positions, node_batch, 
                                    global_features[b], batch_features, data.width[b], data.height[b])
        
        # Update cache
        self.batch_feature_cache = batch_features
        self.batch_input_hash = input_hash
        
        return batch_features


    def _process_bbox(self, bbox, batch_idx, frame_idx, node_features, node_positions, node_batch, 
                    global_feature, batch_features, width, height):
        """
        Process single bounding box and extract features
        
        Args:
            bbox: Bounding box data [x, y, w, h, ...]
            batch_idx: Batch index
            frame_idx: Frame index (0: previous frame, 1: current frame)
            node_features: Node features [num_nodes, feature_dim]
            node_positions: Node positions [num_nodes, 2]
            node_batch: Node batch indices [num_nodes]
            global_feature: Global feature [feature_dim]
            batch_features: Batch features to update [batch_size, n_frames, max_boxes+1, feature_dim]
            width: Image width
            height: Image height
        """
        device = node_features.device
        feature_dim = node_features.shape[1]
        
        # Extract bounding box coordinates
        x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        
        # Get track_id, 6th element (index 5) is track_id
        track_id = int(bbox[5].item()) if bbox.shape[0] > 5 else 1
        
        # Skip if track_id is out of range
        if track_id <= 0 or track_id > self.max_boxes:
            return
        
        # Calculate bounding box coordinates
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Normalize coordinates
        width_val = width.item() if isinstance(width, torch.Tensor) else width
        height_val = height.item() if isinstance(height, torch.Tensor) else height
        
        x1_norm = x1 / width_val
        y1_norm = y1 / height_val
        x2_norm = x2 / width_val
        y2_norm = y2 / height_val
        
        # Calculate bounding box center
        cx_norm, cy_norm = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2
        
        # Find nodes belonging to current batch
        batch_nodes_mask = (node_batch == batch_idx)
        batch_nodes_indices = torch.where(batch_nodes_mask)[0]
        
        if len(batch_nodes_indices) == 0:
            # If no nodes for this batch, use global feature
            batch_features[batch_idx, frame_idx, track_id] = global_feature
            return
        
        # Get node positions for current batch
        batch_nodes_pos = node_positions[batch_nodes_indices]
        
        # Find nodes inside the bounding box
        in_box = (batch_nodes_pos[:, 0] >= x1_norm) & (batch_nodes_pos[:, 0] <= x2_norm) & \
                (batch_nodes_pos[:, 1] >= y1_norm) & (batch_nodes_pos[:, 1] <= y2_norm)
        
        nodes_in_box_indices = batch_nodes_indices[torch.where(in_box)[0]]
        
        if len(nodes_in_box_indices) > 0:
            # Calculate distance from each node to bounding box center
            nodes_pos = node_positions[nodes_in_box_indices]
            center = torch.tensor([cx_norm, cy_norm], device=device)
            distances = torch.norm(nodes_pos - center.unsqueeze(0), dim=1)
            
            # Calculate weights based on distance (closer nodes have higher weights)
            weights = 1.0 / (distances + 1e-6)
            weights = weights / weights.sum()  # Normalize weights
            
            # Weighted average to calculate bounding box feature
            box_feature = (node_features[nodes_in_box_indices] * weights.unsqueeze(1)).sum(dim=0)
        else:
            # If no nodes inside bounding box, find closest nodes
            center = torch.tensor([cx_norm, cy_norm], device=device)
            distances = torch.norm(batch_nodes_pos - center.unsqueeze(0), dim=1)
            
            # Select up to 5 closest nodes
            k = min(5, len(distances))
            closest_indices = torch.topk(distances, k, largest=False)[1]
            closest_nodes = batch_nodes_indices[closest_indices]
            
            if len(closest_nodes) > 0:
                box_feature = node_features[closest_nodes].mean(dim=0)
            else:
                # If no available nodes, use global feature
                box_feature = global_feature
        
        # Store feature
        batch_features[batch_idx, frame_idx, track_id] = box_feature