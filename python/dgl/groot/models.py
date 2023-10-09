import torch
import dgl.backend as F
import torch.nn as nn
from ..nn.pytorch.conv.gatconv import GATConv
from .._ffi.function import _init_api
import dgl.function as fn

_init_api("dgl.groot", __name__)


class Shuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scattered_array, feat, rank, world_size):
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.scattered_array =scattered_array
        out = F.zerocopy_from_dgl_ndarray(\
            _CAPI_ScatterForward(scattered_array,F.zerocopy_to_dgl_ndarray(feat), rank, world_size))
        out.requires_grad = True
        return out
    @staticmethod
    def backward(ctx, grads):
        rank = ctx.rank
        world_size = ctx.world_size
        scattered_array = ctx.scattered_array
        return None,(F.zerocopy_from_dgl_ndarray(\
                _CAPI_ScatterBackward(scattered_array,\
                                      F.zerocopy_to_dgl_ndarray(grads), rank, world_size))),None,None

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
        self.debug = False

    def forward(self, blocks, input, inference = False):
        x = input
        if not inference:
            blocks, frontier, unique_id_list = blocks
        redundant_layer = self.n_layers - self.num_redundant_layers
        if self.debug:
            print(input)
        # Todo refactor this code such that redundant_layer != 0 is handled automatically
        for i, (block,layer) in enumerate(zip(blocks, self.layers)):
            if not inference:
                if i == redundant_layer and redundant_layer != 0:
                    x = Shuffle.apply(frontier, x, self.rank, self.world_size)
                if i <  redundant_layer:
                    x = Shuffle.apply(block.scattered_src, x, self.rank, self.world_size)
            if self.debug:
                print("Layer", i)
                debug(unique_id_list[i],x[:,0],i)
            x = layer(block, x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
            if len(x.shape) == 3:
                x = x.flatten(-2)
        if not inference:
            if len(self.layers) == redundant_layer:
                x = Shuffle.apply(frontier, x, self.rank, self.world_size)
        if self.debug:
            debug( unique_id_list[-1][:x.shape[0]], x[:,0], self.n_layers)
        return x

def debug(unique, values, layer_id):
    _u = unique.tolist()
    _v = values.tolist()
    string = ""
    assert(len(_u) == len(_v))
    for i in range(len(_u)):
        string += f"({_u[i]}:{_v[i]})"
    print("DEBUG !!!!!!!!!!!")
    print(f"{layer_id} : {string}")
    return string

class Gather(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, block, feature):
        with block.local_scope():
            block.srcdata['x'] = feature
            block.update_all(fn.copy_u('x','m'), fn.sum('m', 'h'))
            return block.dstdata['h']

def get_distributed_model(model_type, layer, feat_size, num_layers,\
                            n_hidden, n_classes, num_redundant,\
                                rank, world_size):
    assert(layer == "GAT")
    layers = []
    assert(num_layers > 0)
    heads = 4
    print("n_classes", n_classes)
    # model_type = "debug"
    if model_type == "GAT":
        # GAT has atleast 2 layers
        assert(num_layers > 1)
        layers.append(GATConv(feat_size, n_hidden//heads, heads))
        for i in range(num_layers-2):
            layers.append(GATConv(n_hidden, n_hidden//heads, heads))
        layers.append(GATConv(n_hidden, n_classes, 1))
    if model_type == "test":
        for i in range(num_layers):
            layers.append(Gather())

    model = SRC_TO_DEST(feat_size, n_hidden, n_classes, layers, num_redundant,\
                            rank, world_size)
    return model