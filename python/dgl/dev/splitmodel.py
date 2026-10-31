import torch
import torch.nn as nn
from ..nn.pytorch.conv.gatconv import GATConv
from ..nn.pytorch.conv.hgtconv import HGTConv
from ..nn.pytorch.conv.sageconv import SAGEConv
from ..nn.pytorch.conv.graphconv import GraphConv
from dgl.backend import to_dgl_nd, from_dgl_nd

from .._ffi.function import _init_api
import dgl.function as fn

_init_api("dgl.dev", __name__)

# world_size is actually number of partitions

class Shuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scattered_array, feat, rank, world_size):
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.scattered_array = scattered_array
        out = from_dgl_nd(_CAPI_Split_ScatterForward(scattered_array, to_dgl_nd(feat), rank, world_size))
        out.requires_grad = True
        return out
    
    @staticmethod
    def backward(ctx, grads):
        rank = ctx.rank
        world_size = ctx.world_size
        scattered_array = ctx.scattered_array
        return None, (from_dgl_nd(_CAPI_Split_ScatterBackward(scattered_array, to_dgl_nd(grads), rank, world_size))),None,None

# Todo this architecture that can be improved by
# SRC_TO_DEST is not restricted to SAGE and GCN
# it is applied very generally to HGT as well.
class SRC_TO_DEST(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, layers, \
                 num_dp, rank, world_size, model_name):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = len(layers)
        self.layers = nn.ModuleList()
        for l in layers:
            self.layers.append(l)
        self.num_dp = num_dp
        self.rank = rank
        self.world_size = world_size
        self.activation = torch.nn.ReLU()
        self.debug = False
        self.model_name = model_name

    def forward(self, blocks, input, inference = False):
        x = input
        # Hybrid DP-then-split: the loader builds a ScatteredArray only for the
        # split layers and leaves block.scattered_src=None for the leading DP
        # layers (which, after GetBlocks reverses block order, are the *last*
        # num_dp layers here). Shuffle exactly where a scattered array exists;
        # this is order-independent and reduces to full-split when num_dp==0.
        for i, (block,layer) in enumerate(zip(blocks, self.layers)):
            if not inference and getattr(block, "scattered_src", None) is not None:
                x = Shuffle.apply(block.scattered_src, x, self.rank, self.world_size)
            if(self.model_name == "hgt"):
                x = layer(block, x, torch.zeros(x.shape[0],  dtype = torch.int64, device = x.device) \
                          , torch.zeros(block.num_edges(), dtype = torch.int64, device = x.device))
            else:
                x = layer(block, x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
            if len(x.shape) == 3:
                x = x.flatten(-2)
        return x

# def debug(unique, values, layer_id):
#     _u = unique.tolist()
#     _v = values.tolist()
#     string = ""
#     assert(len(_u) == len(_v))
#     for i in range(len(_u)):
#         string += f"({_u[i]}:{_v[i]})"
#     print("DEBUG !!!!!!!!!!!")
#     print(f"{layer_id} : {string}")
#     return string

# class Gather(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, block, feature):
#         with block.local_scope():
#             block.srcdata['x'] = feature
#             block.update_all(fn.copy_u('x','m'), fn.sum('m', 'h'))
#             return block.dstdata['h']

def get_distributed_model(rank, world_size, num_dp, model_name, feat_size, num_layers, n_hidden, n_classes):
    layers = []
    assert(num_layers > 0)
    heads = 4
    assert model_name in ["sage", "gat", "hgt", "gcn" ]
    if model_name == "gat":
        # GAT has atleast 2 layers
        assert(num_layers > 1)
        layers.append(GATConv(feat_size, n_hidden//heads, heads))
        for i in range(num_layers-2):
            layers.append(GATConv(n_hidden, n_hidden//heads, heads,))
        layers.append(GATConv(n_hidden, n_classes, 1))
    if model_name == "sage":
        assert (num_layers > 1)
        layers.append(SAGEConv(feat_size, n_hidden , aggregator_type= 'mean'))
        for i in range(num_layers - 2):
            layers.append(SAGEConv(n_hidden, n_hidden , aggregator_type= 'mean'))
        layers.append(SAGEConv(n_hidden, n_classes, aggregator_type= 'mean'))
    if model_name == "gcn":
        assert (num_layers > 1)
        layers.append(GraphConv(feat_size, n_hidden))
        for i in range(num_layers - 2):
            layers.append(GraphConv(n_hidden, n_hidden))
        layers.append(GraphConv(n_hidden, n_classes))
    if model_name == "hgt":
        assert (num_layers > 1)
        layers.append(HGTConv(feat_size, n_hidden // heads, heads, num_ntypes=1, num_etypes=1))
        for i in range(num_layers - 2):
            layers.append(HGTConv(n_hidden, n_hidden // heads, heads, num_ntypes= 1, num_etypes = 1))
        layers.append(HGTConv(n_hidden, n_classes, 1, num_ntypes= 1, num_etypes = 1))

    model = SRC_TO_DEST(feat_size, n_hidden, n_classes, layers, num_dp, \
                        rank, world_size, model_name)
    return model