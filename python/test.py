from dgl._ffi.function import _init_api
import dgl.backend as F
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.utils import pin_memory_inplace
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn
import torch.distributed as dist
from torch.nn.functional import cross_entropy
import time

_init_api("dgl.groot", __name__)


class Sage(nn.Module):
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
    
from dgl.heterograph import DGLBlock
def init_dataloader(rank: int, 
                          indptr: torch.Tensor, 
                          indices: torch.Tensor, 
                          feats: torch.Tensor,
                          labels: torch.Tensor,
                          seeds: torch.Tensor,
                          fanouts: list[int],
                          batch_size: int,
                          max_pool_size: int = 2):
    
    return _CAPI_InitLocDataloader(rank, 
                                    F.zerocopy_to_dgl_ndarray(indptr),
                                    F.zerocopy_to_dgl_ndarray(indices),
                                    F.zerocopy_to_dgl_ndarray(feats),
                                    F.zerocopy_to_dgl_ndarray(labels.flatten().to(rank)),
                                    F.zerocopy_to_dgl_ndarray(seeds.to(rank)),
                                    fanouts,
                                    batch_size,
                                    max_pool_size)

def get_batch(key: int, layers:int = 3):
    blocks = []
    for i in range(layers):
        gidx = _CAPI_GetBlock(key, i)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        blocks.insert(0, block)
        
    feat = _CAPI_GetOutputNodeFeats(key)
    labels = _CAPI_GetInputNodeLabels(key)
    return blocks, F.from_dgl_nd(feat), F.from_dgl_nd(labels)


def main(graph_name="ogbn-products"):
    dataset = DglNodePropPredDataset(graph_name, root="/data/sandeep/")
    graph = dataset.graph[0]
    feats = graph.srcdata["feat"]
    train_idx = dataset.get_idx_split()["train"]
    indptr, indices, edge_id = graph.adj_tensors('csc')
    labels = dataset.labels

    feats_handle = pin_memory_inplace(feats)
    indptr_handle = pin_memory_inplace(indptr)
    indices_handle = pin_memory_inplace(indices)
    edge_id_handle = pin_memory_inplace(edge_id)
    # feats = feats.to(0)
    # indptr = indptr.to(0)
    # indices = indices.to(0)
    # edge_id = edge_id.to(0)
    
    batch_size = 1024
    pool_size = 2
    dataloader = init_dataloader(0, indptr, indices, feats, labels, train_idx, [15, 10, 5], batch_size, pool_size)
    model = Sage(in_feats=feats.shape[1], hid_feats=256, num_layers=3, out_feats=dataset.num_classes)
    model = model.to(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = train_idx.shape[0] / batch_size
    step = int(step)
    feat_size_in_bytes = 0
    print(f"start sampling for {step} mini-batches") 
    key_uninitialized = True
    num_epoch = 5
    
    duration_in_s = 0
    for epoch in range(num_epoch):
        start = time.time()
        for i in range(step):
            # if key_uninitialized:
            #     key = _CAPI_NextAsync()
            # blocks, batch_feat, batch_labels = get_batch(key)
            # key = _CAPI_NextAsync()
            # key_uninitialized = False
            key = _CAPI_NextSync()
            blocks, batch_feat, batch_labels = get_batch(key)

            pred = model(blocks, batch_feat)
            loss = cross_entropy(pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_feat, feat_width = batch_feat.shape
            feat_size_in_bytes += num_feat * feat_width * 4
            
            if key % 50 == 0:
                print(f"{key=}")
        end = time.time()
        if epoch > 0:
            duration_in_s += end - start
    feat_size_in_mb = feat_size_in_bytes / 1024 / 1024
    per_epoch_duration = duration_in_s / (num_epoch - 1)
    print(f"finished sampling one epoch in {round(per_epoch_duration, 2)} sec")
    print(f"fetching feature data {round(feat_size_in_mb)} MB; bandwidth {round(feat_size_in_mb / duration_in_s)} MB/s")
    
if __name__ == "__main__":
    main()