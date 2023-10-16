# Contruct a n-layer GNN model
from dgl.nn.pytorch.conv import GATConv, SAGEConv
import torch.nn as nn
import torch
class DGLGraphSage(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, num_layers: int, out_feats: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        self.fwd_l1_timer = []    
        self.hid_feats_lst = []
        
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
            elif layer_idx >= 1 and layer_idx < num_layers - 1:            
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            else:
                # last layer
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))

    def forward(self, blocks, feat):
        hid_feats = feat
        l1_start = torch.cuda.Event(enable_timing=True)
        l1_start.record()
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if (layer_idx == 0):
                l1_end = torch.cuda.Event(enable_timing=True)
                l1_end.record()
                self.fwd_l1_timer.append((l1_start, l1_end))   
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
        return hid_feats
    
    def fwd_l1_time(self):
        torch.cuda.synchronize()
        fwd_time = 0.0
        for l1_start, l1_end in self.fwd_l1_timer:
            fwd_time += l1_start.elapsed_time(l1_end)
        self.fwd_l1_timer = []
        return fwd_time
    
class DGLGat(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, num_layers: int, out_feats: int, num_heads: int=4):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        self.fwd_l1_timer = []    
        self.hid_feats_lst = []
        hid_feats = int(hid_feats/num_heads)
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(GATConv(in_feats=in_feats, out_feats=hid_feats, num_heads=num_heads))
            elif layer_idx >= 1 and layer_idx < num_layers - 1:            
                self.layers.append(GATConv(
                    in_feats=hid_feats * num_heads, out_feats=hid_feats, num_heads=num_heads))
            else:
                # last layer
                self.layers.append(GATConv(
                    in_feats=hid_feats * num_heads, out_feats=out_feats, num_heads=1))

    def forward(self, blocks, feat, inference = False):
        hid_feats = feat
        l1_start = torch.cuda.Event(enable_timing=True)
        l1_start.record()
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if (layer_idx == 0):
                l1_end = torch.cuda.Event(enable_timing=True)
                l1_end.record()
                self.fwd_l1_timer.append((l1_start, l1_end))   
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
            hid_feats = hid_feats.flatten(1)
        return hid_feats
    
    def fwd_l1_time(self):
        torch.cuda.synchronize()
        fwd_time = 0.0
        for l1_start, l1_end in self.fwd_l1_timer:
            fwd_time += l1_start.elapsed_time(l1_end)
        self.fwd_l1_timer = []
        return fwd_time
    
# def get_dgl_model(config: RunConfig) -> nn.Module:
#     if config.model_type == "graphsage":
#         return DGLGraphSage(in_feats=config.in_feat, 
#                              hid_feats=config.hid_feat,
#                              num_layers=len(config.fanouts),
#                              out_feats=config.num_classes)
#     elif config.model_type == "gat":
#         return DGLGat(in_feats=config.in_feat, 
#                         hid_feats=config.hid_feat,
#                         num_layers=len(config.fanouts),
#                         out_feats=config.num_classes,
#                         num_heads=4)