import torch
import dgl.backend as F
import torch.nn as nn
from ..nn.pytorch.conv.gatconv import GATConv
from .._ffi.function import _init_api
_init_api("dgl.groot", __name__)

class Shuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scattered_array, feat, rank, world_size):
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.scattered_array =scattered_array
        return F.zerocopy_from_dgl_ndarray(\
            _CAPI_ScatterForward(scattered_array,F.zerocopy_to_dgl_ndarray(feat), rank, world_size))

    @staticmethod
    def backward(ctx, grads):
        rank = ctx.rank
        world_size = ctx.world_size
        scattered_array = ctx.scattered_array
        return (F.zerocopy_from_dgl_ndarray(\
                _CAPI_ScatterBackward(scattered_array,\
                                      F.zerocopy_to_dgl_ndarray(grads), rank, world_size)))

# Todo this architecture that can be improved by
# SRC_TO_DEST is not restricted to SAGE and GCN
# it is applied very generally to HGT as well.
class SRC_TO_DEST(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, layers,\
                            num_redundant_layers, rank, world_size):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = len(layers)
        self.layers = nn.ModuleList()
        for l in layers:
            self.layers.append(l)
        self.num_redundant_layers = num_redundant_layers
        self.rank = rank
        self.world_size = world_size
        self.activation = torch.nn.ReLU()

    def forward(self, blocks, input):
        x = input
        blocks, frontier = blocks
        redundant_layer = self.n_layers - self.num_redundant_layers
        for i, (block,layer) in enumerate(zip(blocks, self.layers)):
            #     // switch frontier
            print("running layer", block, i)
            if i == redundant_layer:
                x = Shuffle.apply(frontier, x, self.rank, self.world_size)
            if i <  redundant_layer:
                x = Shuffle.apply(block.scattered_src, x, self.rank, self.world_size)
            x = layer(block, x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
            if len(x.shape) == 3:
                x = x.flatten(-2)
        return x
def get_distributed_model(mode, layer, feat_size, num_layers,\
                            n_hidden, n_classes, num_redundant,\
                                rank, world_size):
    assert(mode == "SRC_TO_DEST")
    assert(layer == "GAT")
    layers = []
    assert(num_layers > 1)
    heads = 4
    print("Adding zero in degree")
    layers.append(GATConv(feat_size, n_hidden//heads, heads, allow_zero_in_degree = True))
    for i in range(num_layers-2):
        layers.append(GATConv(n_hidden, n_hidden//heads, heads, allow_zero_in_degree= True))
    layers.append(GATConv(n_hidden, n_classes, 1, allow_zero_in_degree=  True))
    model = SRC_TO_DEST(feat_size, n_hidden, n_classes, layers, num_redundant,\
                            rank, world_size)
    return model