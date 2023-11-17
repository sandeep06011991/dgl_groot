from dgl.nn.pytorch.conv import GATConv, SAGEConv
import torch.nn as nn
import torch
import torch.distributed as dist
from .util import *

class P3Shuffle(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def forward(ctx, 
                self_rank: int, 
                world_size:int,
                local_hid: torch.Tensor,
                local_hids: list[torch.Tensor],
                global_grads: list[torch.Tensor])->torch.Tensor:
        # print(f"forward {self_rank=} {world_size=} {local_hid.shape}")        
        ctx.self_rank = self_rank
        ctx.world_size = world_size
        ctx.global_grads = global_grads
        # aggregated_hid = torch.clone(local_hid)
        aggregated_hid = local_hid.detach().clone()
        handle = None
        for r in range(world_size):
            if r == self_rank:
                handle = dist.reduce(tensor=aggregated_hid, dst=r, async_op=True) # gathering data from other GPUs
            else:
                dist.reduce(tensor=local_hids[r], dst=r, async_op=True) # TODO: Async gathering data from other GPUs
        handle.wait()
        return aggregated_hid
    
    @staticmethod
    def backward(ctx, grad_outputs):
        # print(f"self.rank={ctx.self_rank} send_grad_shape={grad_outputs.shape} global_grads_shape={[x.shape for x in ctx.global_grads]}")
        dist.all_gather(tensor=grad_outputs, tensor_list=ctx.global_grads)
        return None, None, grad_outputs, None, None

class GatP3First(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, num_heads: int):
        super().__init__()
        self.conv = GATConv(in_feats=in_feats, out_feats=int(hid_feats / num_heads), num_heads=num_heads)
        
    def forward(self, block, feat):
        return self.conv(block, feat).flatten(1)
    
    
class GatP3(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, num_layers: int, out_feats: int, num_heads: int=4):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        self.hid_feats_lst = []
        hid_feats = int(hid_feats/num_heads)
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                continue
            elif layer_idx >= 1 and layer_idx < num_layers - 1:            
                self.layers.append(GATConv(
                    in_feats=hid_feats * num_heads, out_feats=hid_feats, num_heads=num_heads))
            else:
                # last layer
                self.layers.append(GATConv(
                    in_feats=hid_feats * num_heads, out_feats=out_feats, num_heads=1))

    def forward(self, blocks, feat):
        hid_feats = feat
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
            hid_feats = hid_feats.flatten(1)
        return hid_feats
    
class SageP3(nn.Module):
    def __init__(self, 
                 in_feats: int,
                 hid_feats: int, 
                 num_layers: int, 
                 out_feats: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # first layer
                continue
                # self.layers.append(P3_SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
            elif layer_idx >= 1 and layer_idx < num_layers - 1:          
                # middle layers  
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            else:
                # last layer
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))

    def forward(self, blocks, feat):
        hid_feats = feat
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
        return hid_feats
    
def create_gat_p3(rank:int | torch.cuda.device, in_feats:int, hid_feats:int, num_classes:int, num_layers: int, num_heads: int=4) -> tuple[nn.Module, nn.Module]:
    first_layer = GatP3First(in_feats, hid_feats, num_heads).to(rank) # Intra-Model Parallel
    remain_layers = GatP3(in_feats, hid_feats, num_layers, num_classes, num_heads=num_heads).to(rank) # Data Parallel
    return (first_layer, remain_layers)

def create_sage_p3(rank:int | torch.cuda.device, in_feats:int, hid_feats:int, num_classes:int, num_layers: int) -> tuple[nn.Module, nn.Module]:
    first_layer = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean").to(rank) # Intra-Model Parallel
    remain_layers = SageP3(in_feats, hid_feats, num_layers, num_classes).to(rank) # Data Parallel
    return (first_layer, remain_layers)

def get_p3_model(config: Config):
    print(f"get_p3_model: {config=}")
    device = torch.cuda.current_device()
    if config.model == "sage":
        return create_sage_p3(device, config.in_feat, config.hid_size, config.num_classes, len(config.fanouts))
    elif config.model == "gat":
        return create_gat_p3(device, config.in_feat, config.hid_size, config.num_classes, len(config.fanouts))
    else:
        print(f"invalid model type {config.model}")
        exit(-1) 

