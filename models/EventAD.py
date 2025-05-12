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
        self.event_proj = nn.Linear(event_dim, 128)
        self.coord_proj = nn.Linear(coord_dim, 128)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, 4)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Output 2 classes: normal and anomaly
        )
    
    def forward(self, event_feature, coord_feature):
        """
        Fuse event features and coordinate features
        
        Args:
            event_feature: Event features [batch_size, event_dim]
            coord_feature: Coordinate features [batch_size, coord_dim]
            
        Returns:
            output: Fused features [batch_size, 2]
        """
        # Projection
        event_proj = self.event_proj(event_feature)  # [batch_size, 128]
        coord_proj = self.coord_proj(coord_feature)  # [batch_size, 128]
        
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
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, query, key, value):
        """
        Spatial attention mechanism
        
        Args:
            query: Query tensor [batch_size, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]
            value: Value tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            context: Context vector [batch_size, hidden_dim]
        """
        # Check and process input dimensions
        if query.dim() == 4:  # [batch, channels, height, width]
            # Convert 4D tensor to 2D
            batch_size = query.size(0)
            query = query.view(batch_size, -1)  # [batch, channels*height*width]
        
        if key.dim() == 4:
            # Convert 4D tensor to 3D
            batch_size = key.size(0)
            seq_len = key.size(1)
            key = key.view(batch_size, seq_len, -1)  # [batch, seq_len, channels*height*width]
        
        if value.dim() == 4:
            # Convert 4D tensor to 3D
            batch_size = value.size(0)
            seq_len = value.size(1)
            value = value.view(batch_size, seq_len, -1)  # [batch, seq_len, channels*height*width]
        
        # Ensure query is a 2D tensor [batch_size, hidden_dim]
        if query.dim() == 3 and query.size(1) == 1:
            query = query.squeeze(1)
        
        # Calculate attention scores
        # [batch_size, 1, hidden_dim] x [batch_size, hidden_dim, seq_len] = [batch_size, 1, seq_len]
        scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(1)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)
        
        # Calculate context vector
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim]
        context = torch.bmm(weights.unsqueeze(1), value).squeeze(1)
        
        return context
    

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
        self.soft_attention = SpatialAttention()
        self.soft_attention_cor = SpatialAttention()
        
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
            testing: Whether in test mode
            
        Returns:
            losses: Loss dictionary
            outputs: Output list
            output_labels: Output labels list
        """
        batch_size = data.num_graphs
        device = data.x.device
        
        # Extract features
        batch_features = self.extract_features(data)
        
        # Initialize output and label lists
        batch_outputs = []
        batch_labels = []
        
        # Process each batch
        for b in range(batch_size):
            frame_outputs = []
            frame_labels = []
            
            # Get features for the current batch
            prev_features = batch_features[b, 0]  # Previous frame features
            curr_features = batch_features[b, 1]  # Current frame features
            
            # Process each bounding box
            for track_id in range(1, self.max_boxes + 1):
                # Get features for the previous and current frames
                prev_feature = prev_features[track_id]
                curr_feature = curr_features[track_id]
                
                # Check if features are valid (non-zero)
                if torch.sum(prev_feature) == 0 or torch.sum(curr_feature) == 0:
                    continue
                
                # Expand features into sequence [1, 2, feature_dim]
                event_seq = torch.stack([prev_feature, curr_feature]).unsqueeze(0)
                
                # Get corresponding bounding box coordinates
                # Find bounding box for current batch and track_id
                if hasattr(data, 'bbox') and data.bbox is not None:
                    bbox_indices = torch.where((data.bbox_batch == b) & (data.bbox[:, 5] == track_id))[0]
                    if len(bbox_indices) > 0:
                        bbox_idx = bbox_indices[0]
                        bbox_data = data.bbox[bbox_idx]
                        
                        # Extract coordinates [x, y, w, h]
                        coord = bbox_data[:4].float().unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                    else:
                        # If no corresponding bounding box is found, skip
                        continue
                else:
                    # If no bounding box data, skip
                    continue
                
                # Initialize hidden states
                h_event = torch.zeros(self.gru_net_event.n_layers, 1, self.gru_net_event.hidden_dim, device=device)
                h_coord = torch.zeros(self.gru_net_cor.n_layers, 1, self.gru_net_cor.hidden_dim, device=device)
                
                # Use GRU to process event feature sequence
                event_output, _ = self.gru_net_event(event_seq, h_event)  # [1, 2, hidden_dim]
                # Take output from the last time step
                event_output = event_output[:, -1, :]  # [1, hidden_dim]
                
                # Use GRU to process coordinate sequence
                coord_output, _ = self.gru_net_cor(coord, h_coord)  # [1, 1, 32]
                # Take output from the last time step
                coord_output = coord_output[:, -1, :]  # [1, 32]
                
                # Fuse features
                output = self.fusion_module(event_output, coord_output)  # [1, 2]
                
                # Add to output list
                frame_outputs.append(output.squeeze(0))
                
                # If labels are provided, add to labels list
                if labels is not None:
                    # Find corresponding label
                    if hasattr(data, 'bbox') and data.bbox is not None:
                        bbox_indices = torch.where((data.bbox_batch == b) & (data.bbox[:, 5] == track_id))[0]
                        if len(bbox_indices) > 0:
                            bbox_idx = bbox_indices[0]
                            label = labels[bbox_idx].long()
                            frame_labels.append(label)
                        else:
                            # If no corresponding label is found, use default value 0
                            frame_labels.append(torch.tensor(0, device=device).long())
                    else:
                        # If no bounding box data, use default value 0
                        frame_labels.append(torch.tensor(0, device=device).long())
            
            # If the current batch has outputs, add to batch output list
            if len(frame_outputs) > 0:
                batch_outputs.append(frame_outputs)
                batch_labels.append(frame_labels)
        
        # Calculate loss
        losses = {}
        if labels is not None and len(batch_outputs) > 0:
            # Collect all predictions and labels
            all_outputs = []
            all_labels = []
            
            for frame_outputs, frame_labels in zip(batch_outputs, batch_labels):
                all_outputs.extend(frame_outputs)
                all_labels.extend(frame_labels)
            
            if len(all_outputs) > 0:
                # Convert lists to tensors
                all_outputs = torch.stack(all_outputs)
                all_labels = torch.stack(all_labels)
                
                # Calculate cross entropy loss
                loss = self.ce_loss(all_outputs, all_labels)
                losses['cross_entropy'] = loss
            else:
                # If no valid predictions, use zero loss
                losses['cross_entropy'] = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # If no labels or outputs are empty, use zero loss
            losses['cross_entropy'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        return losses, batch_outputs, batch_labels
    
    def extract_features(self, data):
        """
        Extract features
        
        Args:
            data: Input data
            
        Returns:
            batch_features: Batch features [batch_size, n_frames, max_boxes, feature_dim]
        """
        batch_size = data.num_graphs
        device = data.x.device
        
        # Calculate input hash for caching, handling possible None cases
        x_sum = data.x.sum().item() if hasattr(data, 'x') and data.x is not None else 0
        edge_sum = data.edge_index.sum().item() if hasattr(data, 'edge_index') and data.edge_index is not None else 0
        input_hash = hash(str(x_sum) + str(edge_sum))
        
        # If cache is valid, return directly
        if self.batch_feature_cache is not None and self.batch_input_hash == input_hash:
            return self.batch_feature_cache
        
        # Use DAGR model to extract node features and global features
        with torch.no_grad():  # Don't calculate gradients for DAGR model
            self.dagr_model.eval()  # Ensure DAGR model is in evaluation mode
            try:
                node_features, global_feature = self.dagr_model.extract_features(data)
            except Exception as e:
                print(f"Error extracting features with DAGR model: {e}")
                # If DAGR model fails to extract features, create random features as fallback
                # Assume feature dimension is 256
                feature_dim = 256
                node_features = torch.randn((data.num_nodes, feature_dim), device=device)
                global_feature = torch.randn((batch_size, feature_dim), device=device)
        
        # Get node positions and batch indices
        node_positions = data.pos if hasattr(data, 'pos') and data.pos is not None else torch.zeros((0, 2), device=device)
        node_batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(0, dtype=torch.long, device=device)
        
        # Initialize batch features
        # [batch_size, n_frames, max_boxes, feature_dim]
        feature_dim = node_features.shape[1] if node_features.size(0) > 0 else 256
        batch_features = torch.zeros(
            (batch_size, 2, self.max_boxes + 1, feature_dim), 
            device=device
        )
        
        # Process each batch
        for b in range(batch_size):
            # Extract single sample data
            sample_data = self._extract_sample_from_batch(data, b)
            
            # Get image width and height
            width = sample_data.width if hasattr(sample_data, 'width') else 640
            height = sample_data.height if hasattr(sample_data, 'height') else 480
            
            # Process previous frame bounding boxes (bbox0)
            if hasattr(sample_data, 'bbox0') and sample_data.bbox0 is not None:
                try:
                    # Process single bounding box
                    if isinstance(sample_data.bbox0, torch.Tensor):
                        if sample_data.bbox0.dim() == 1:  # Single bounding box
                            bbox_data = sample_data.bbox0
                            
                            # Extract bounding box coordinates
                            x = float(bbox_data[0].item())
                            y = float(bbox_data[1].item())
                            w = float(bbox_data[2].item())
                            h = float(bbox_data[3].item())
                            
                            # Get track_id, the last element is track_id
                            track_id = int(bbox_data[5].item()) if bbox_data.shape[0] > 5 else 1
                            
                            # Calculate bounding box coordinates
                            x1, y1 = x, y
                            x2, y2 = x + w, y + h
                            
                            # Normalize coordinates
                            x1_norm = x1 / width if isinstance(width, (int, float)) else x1 / width.item()
                            y1_norm = y1 / height if isinstance(height, (int, float)) else y1 / height.item()
                            x2_norm = x2 / width if isinstance(width, (int, float)) else x2 / width.item()
                            y2_norm = y2 / height if isinstance(height, (int, float)) else y2 / height.item()
                            
                            # Find all nodes within the bounding box
                            if node_positions.size(0) > 0:
                                in_box = (node_positions[:, 0] >= x1_norm) & (node_positions[:, 0] <= x2_norm) & \
                                        (node_positions[:, 1] >= y1_norm) & (node_positions[:, 1] <= y2_norm)
                                if node_batch.size(0) > 0:
                                    in_box = in_box & (node_batch == b)  # Ensure only nodes from current batch are considered
                            else:
                                in_box = torch.zeros(0, dtype=torch.bool, device=device)
                            
                            # Get bounding box center
                            cx_norm, cy_norm = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2
                            
                            # Find nodes inside the bounding box
                            nodes_in_box = torch.where(in_box)[0]
                            
                            # If there are nodes in the box, calculate weighted average feature
                            if len(nodes_in_box) > 0:
                                # Calculate distance from each node to box center
                                nodes_pos = node_positions[nodes_in_box, :2]
                                center = torch.tensor([cx_norm, cy_norm], device=device)
                                distances = torch.norm(nodes_pos - center.unsqueeze(0), dim=1)
                                
                                # Calculate weights based on distance (closer nodes get higher weights)
                                weights = 1.0 / (distances + 1e-6)
                                weights = weights / weights.sum()  # Normalize
                                
                                # Weighted average
                                box_feature = (node_features[nodes_in_box] * weights.unsqueeze(1)).sum(dim=0)
                            else:
                                # If no nodes in the box, use the 5 closest nodes to the box center
                                batch_nodes = torch.where(node_batch == b)[0] if node_batch.size(0) > 0 else torch.tensor([], dtype=torch.long, device=device)
                                if len(batch_nodes) > 0:
                                    distances = torch.norm(node_positions[batch_nodes, :2] - torch.tensor([cx_norm, cy_norm], device=device), dim=1)
                                    closest_nodes = batch_nodes[torch.argsort(distances)[:min(5, len(batch_nodes))]]
                                    box_feature = node_features[closest_nodes].mean(dim=0)
                                else:
                                    # If no nodes, use global feature
                                    box_feature = global_feature[b] if global_feature.size(0) > b else torch.zeros(feature_dim, device=device)
                            
                            # Store feature
                            if 0 < track_id <= self.max_boxes:
                                batch_features[b, 0, track_id] = box_feature
                        
                        elif sample_data.bbox0.dim() == 2:  # Multiple bounding boxes
                            for i in range(sample_data.bbox0.shape[0]):
                                bbox_data = sample_data.bbox0[i]
                                
                                # Extract bounding box coordinates
                                x = float(bbox_data[0].item())
                                y = float(bbox_data[1].item())
                                w = float(bbox_data[2].item())
                                h = float(bbox_data[3].item())
                                
                                # Get track_id, the last element is track_id
                                track_id = int(bbox_data[5].item()) if bbox_data.shape[0] > 5 else i + 1
                                
                                # Calculate bounding box coordinates
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h
                                
                                # Normalize coordinates
                                x1_norm = x1 / width if isinstance(width, (int, float)) else x1 / width.item()
                                y1_norm = y1 / height if isinstance(height, (int, float)) else y1 / height.item()
                                x2_norm = x2 / width if isinstance(width, (int, float)) else x2 / width.item()
                                y2_norm = y2 / height if isinstance(height, (int, float)) else y2 / height.item()
                                
                                # Find all nodes within the bounding box
                                in_box = (node_positions[:, 0] >= x1_norm) & (node_positions[:, 0] <= x2_norm) & \
                                        (node_positions[:, 1] >= y1_norm) & (node_positions[:, 1] <= y2_norm) & \
                                        (node_batch == b)  # Ensure only nodes from current batch are considered
                                
                                # Get bounding box center
                                cx_norm, cy_norm = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2
                                
                                # Find nodes inside the bounding box
                                nodes_in_box = torch.where(in_box)[0]
                                
                                # If there are nodes in the box, calculate weighted average feature
                                if len(nodes_in_box) > 0:
                                    # Calculate distance from each node to box center
                                    nodes_pos = node_positions[nodes_in_box, :2]
                                    center = torch.tensor([cx_norm, cy_norm], device=device)
                                    distances = torch.norm(nodes_pos - center.unsqueeze(0), dim=1)
                                    
                                    # Calculate weights based on distance (closer nodes get higher weights)
                                    weights = 1.0 / (distances + 1e-6)
                                    weights = weights / weights.sum()  # Normalize
                                    
                                    # Weighted average
                                    box_feature = (node_features[nodes_in_box] * weights.unsqueeze(1)).sum(dim=0)
                                else:
                                    # If no nodes in the box, use the 5 closest nodes to the box center
                                    batch_nodes = torch.where(node_batch == b)[0]
                                    if len(batch_nodes) > 0:
                                        distances = torch.norm(node_positions[batch_nodes, :2] - torch.tensor([cx_norm, cy_norm], device=device), dim=1)
                                        # Ensure index doesn't go out of bounds
                                        closest_nodes = batch_nodes[torch.argsort(distances)[:min(5, len(batch_nodes))]]
                                        box_feature = node_features[closest_nodes].mean(dim=0)
                                    else:
                                        # If no nodes, use global feature
                                        box_feature = global_feature
                                
                                # Store feature
                                if 0 < track_id <= self.max_boxes:
                                    batch_features[b, 0, track_id] = box_feature
                except Exception as e:
                    print(f"Error processing previous frame bounding boxes: {e}")
            
            # Process current frame bounding boxes (bbox)
            if hasattr(sample_data, 'bbox') and sample_data.bbox is not None:
                try:
                    # Process single bounding box
                    if isinstance(sample_data.bbox, torch.Tensor):
                        if sample_data.bbox.dim() == 1:  # Single bounding box [x, y, w, h, ?, track_id]
                            bbox_data = sample_data.bbox
                            
                            # Extract bounding box coordinates
                            x = float(bbox_data[0].item())
                            y = float(bbox_data[1].item())
                            w = float(bbox_data[2].item())
                            h = float(bbox_data[3].item())
                            
                            # Get track_id, the last element is track_id
                            track_id = int(bbox_data[5].item()) if bbox_data.shape[0] > 5 else 1
                            
                            # Calculate bounding box coordinates
                            x1, y1 = x, y
                            x2, y2 = x + w, y + h
                            
                            # Normalize coordinates
                            x1_norm = x1 / width if isinstance(width, (int, float)) else x1 / width.item()
                            y1_norm = y1 / height if isinstance(height, (int, float)) else y1 / height.item()
                            x2_norm = x2 / width if isinstance(width, (int, float)) else x2 / width.item()
                            y2_norm = y2 / height if isinstance(height, (int, float)) else y2 / height.item()
                            
                            # Find all nodes within the bounding box
                            in_box = (node_positions[:, 0] >= x1_norm) & (node_positions[:, 0] <= x2_norm) & \
                                    (node_positions[:, 1] >= y1_norm) & (node_positions[:, 1] <= y2_norm) & \
                                    (node_batch == b)  # Ensure only nodes from current batch are considered
                            
                            # Get bounding box center
                            cx_norm, cy_norm = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2
                            
                            # Find nodes inside the bounding box
                            nodes_in_box = torch.where(in_box)[0]
                            
                            # If there are nodes in the box, calculate weighted average feature
                            if len(nodes_in_box) > 0:
                                # Calculate distance from each node to box center
                                nodes_pos = node_positions[nodes_in_box, :2]
                                center = torch.tensor([cx_norm, cy_norm], device=device)
                                distances = torch.norm(nodes_pos - center.unsqueeze(0), dim=1)
                                
                                # Calculate weights based on distance (closer nodes get higher weights)
                                weights = 1.0 / (distances + 1e-6)
                                weights = weights / weights.sum()  # Normalize
                                
                                # Weighted average
                                box_feature = (node_features[nodes_in_box] * weights.unsqueeze(1)).sum(dim=0)
                            else:
                                # If no nodes in the box, use the 5 closest nodes to the box center
                                batch_nodes = torch.where(node_batch == b)[0]
                                if len(batch_nodes) > 0:
                                    distances = torch.norm(node_positions[batch_nodes, :2] - torch.tensor([cx_norm, cy_norm], device=device), dim=1)
                                    # Ensure index doesn't go out of bounds
                                    closest_nodes = batch_nodes[torch.argsort(distances)[:min(5, len(batch_nodes))]]
                                    box_feature = node_features[closest_nodes].mean(dim=0)
                                else:
                                    # If no nodes, use global feature
                                    box_feature = global_feature
                            
                            # Store feature
                            if 0 < track_id <= self.max_boxes:
                                batch_features[b, 1, track_id] = box_feature
                        
                        elif sample_data.bbox.dim() == 2:  # Multiple bounding boxes
                            for i in range(sample_data.bbox.shape[0]):
                                bbox_data = sample_data.bbox[i]
                                
                                # Extract bounding box coordinates
                                x = float(bbox_data[0].item())
                                y = float(bbox_data[1].item())
                                w = float(bbox_data[2].item())
                                h = float(bbox_data[3].item())
                                
                                # Get track_id, the last element is track_id
                                # Add boundary check to ensure index doesn't go out of bounds
                                track_id = int(bbox_data[5].item()) if bbox_data.shape[0] > 5 else i + 1
                                
                                # Calculate bounding box coordinates
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h
                                
                                # Normalize coordinates
                                x1_norm = x1 / width if isinstance(width, (int, float)) else x1 / width.item()
                                y1_norm = y1 / height if isinstance(height, (int, float)) else y1 / height.item()
                                x2_norm = x2 / width if isinstance(width, (int, float)) else x2 / width.item()
                                y2_norm = y2 / height if isinstance(height, (int, float)) else y2 / height.item()
                                
                                # Find all nodes within the bounding box
                                in_box = (node_positions[:, 0] >= x1_norm) & (node_positions[:, 0] <= x2_norm) & \
                                        (node_positions[:, 1] >= y1_norm) & (node_positions[:, 1] <= y2_norm) & \
                                        (node_batch == b)  # Ensure only nodes from current batch are considered
                                
                                # Get bounding box center
                                cx_norm, cy_norm = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2
                                
                                # Find nodes inside the bounding box
                                nodes_in_box = torch.where(in_box)[0]
                                
                                # If there are nodes in the box, calculate weighted average feature
                                if len(nodes_in_box) > 0:
                                    # Calculate distance from each node to box center
                                    nodes_pos = node_positions[nodes_in_box, :2]
                                    center = torch.tensor([cx_norm, cy_norm], device=device)
                                    distances = torch.norm(nodes_pos - center.unsqueeze(0), dim=1)
                                    
                                    # Calculate weights based on distance (closer nodes get higher weights)
                                    weights = 1.0 / (distances + 1e-6)
                                    weights = weights / weights.sum()  # Normalize
                                    
                                    # Weighted average
                                    box_feature = (node_features[nodes_in_box] * weights.unsqueeze(1)).sum(dim=0)
                                else:
                                    # If no nodes in the box, use the 5 closest nodes to the box center
                                    batch_nodes = torch.where(node_batch == b)[0]
                                    if len(batch_nodes) > 0:
                                        distances = torch.norm(node_positions[batch_nodes, :2] - torch.tensor([cx_norm, cy_norm], device=device), dim=1)
                                        # Ensure index doesn't go out of bounds
                                        closest_nodes = batch_nodes[torch.argsort(distances)[:min(5, len(batch_nodes))]]
                                        box_feature = node_features[closest_nodes].mean(dim=0)
                                    else:
                                        # If no nodes, use global feature
                                        box_feature = global_feature
                                
                                # Store feature
                                if 0 < track_id <= self.max_boxes:
                                    batch_features[b, 1, track_id] = box_feature
                except Exception as e:
                    print(f"Error processing current frame bounding boxes: {e}")
        
        # Update cache
        self.batch_feature_cache = batch_features
        self.batch_input_hash = input_hash
        
        return batch_features

    def _extract_sample_from_batch(self, data, batch_idx):
        """
        Extract a single sample from batch data
        
        Args:
            data: Batch data
            batch_idx: Batch index
            
        Returns:
            sample_data: Single sample data
        """
        # Create a new data object
        from torch_geometric.data import Data
        sample_data = Data()
        
        # Copy basic attributes
        for key in data.keys():
            if not hasattr(data, key):
                continue
                
            attr = getattr(data, key)
            
            # Skip None values
            if attr is None:
                continue
                
            # Process different types of attributes
            if key == 'batch':
                # Batch index doesn't need to be copied
                continue
            elif key == 'pos' or key == 'x':
                # For node attributes, filter by batch index
                if hasattr(data, 'batch') and data.batch is not None:
                    batch_mask = (data.batch == batch_idx)
                    setattr(sample_data, key, attr[batch_mask])
                else:
                    # If no batch index, assume there's only one sample
                    setattr(sample_data, key, attr)
            elif key == 'edge_index' or key == 'edge_attr':
                # For edge attributes, node indices need to be remapped
                if hasattr(data, 'batch') and data.batch is not None:
                    # Get node mask for current batch
                    batch_mask = (data.batch == batch_idx)
                    
                    # Get node index mapping for current batch
                    node_idx = torch.zeros(data.num_nodes, dtype=torch.long, device=data.pos.device)
                    node_idx[batch_mask] = torch.arange(batch_mask.sum(), device=data.pos.device)
                    
                    # Filter edges for current batch
                    if key == 'edge_index':
                        # Find edges where both endpoints are in the current batch
                        edge_mask = batch_mask[attr[0]] & batch_mask[attr[1]]
                        # Remap node indices
                        new_edge_index = node_idx[attr[:, edge_mask]]
                        setattr(sample_data, key, new_edge_index)
                    elif key == 'edge_attr':
                        # Edge attributes correspond to edge_index
                        if hasattr(data, 'edge_index'):
                            edge_mask = batch_mask[data.edge_index[0]] & batch_mask[data.edge_index[1]]
                            setattr(sample_data, key, attr[edge_mask])
                else:
                    # If no batch index, assume there's only one sample
                    setattr(sample_data, key, attr)
            elif key == 'bbox' or key == 'bbox0':
                # For bounding boxes, filter by batch index
                if isinstance(attr, list):
                    # If it's a list, each element corresponds to a batch
                    if batch_idx < len(attr):
                        setattr(sample_data, key, attr[batch_idx])
                elif isinstance(attr, torch.Tensor) and attr.dim() > 1:
                    # If it's a tensor, the first dimension might be batch
                    if attr.shape[0] > batch_idx:
                        setattr(sample_data, key, attr[batch_idx])
                else:
                    # Otherwise, assume all bounding boxes belong to the same batch
                    setattr(sample_data, key, attr)
            elif key == 'image':
                # For images, filter by batch index
                if attr.dim() >= 4 and attr.shape[0] > batch_idx:
                    setattr(sample_data, key, attr[batch_idx])
            elif isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] > batch_idx:
                # For other tensor attributes, if first dimension is greater than batch index, filter by batch index
                setattr(sample_data, key, attr[batch_idx])
            else:
                # Other attributes are copied directly
                setattr(sample_data, key, attr)
        
        # Ensure sample data has necessary attributes
        if not hasattr(sample_data, 'pos') and hasattr(data, 'pos'):
            # If no pos attribute, create an empty one
            sample_data.pos = torch.zeros((0, data.pos.shape[1]), device=data.pos.device)
        
        # Add width and height attributes (if original data has them)
        if hasattr(data, 'width'):
            if isinstance(data.width, torch.Tensor) and data.width.numel() > batch_idx:
                sample_data.width = data.width[batch_idx]
            else:
                sample_data.width = data.width
        
        if hasattr(data, 'height'):
            if isinstance(data.height, torch.Tensor) and data.height.numel() > batch_idx:
                sample_data.height = data.height[batch_idx]
            else:
                sample_data.height = data.height
        
        return sample_data
