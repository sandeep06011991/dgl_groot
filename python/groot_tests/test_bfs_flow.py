from ..runner.util import *

def get_test_dataset():
    print("reading synthetic ")
    clique_size = 16
    e_list = []
    for i in range(clique_size):
        for j in range(4):
            #     if i == j:
            #         continue
            e_list.append([i, (i + 1 + j) % clique_size])
    graph = dgl.DGLGraph(e_list)
    label = torch.zeros(clique_size).to(torch.int64)
    graph.ndata['feat'] = \
        (torch.arange(graph.num_nodes()) * torch.ones((2, graph.num_nodes()), dtype=torch.float32)).T.contiguous()
    print(graph.ndata['feat'].shape)
    graph.ndata['label'] = label
    dataset_idx_split = {}
    for i in ["train", "test", "valid"]:
        dataset_idx_split[i] = torch.arange(clique_size)
    class Dataset:
        pass
    dataset = Dataset()
    dataset.num_classes = 1


import torch, dgl
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from ..runner.util import *
from ..dgl.groot import *
from ..dgl.utils import pin_memory_inplace
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from ..dgl.groot.models import get_distributed_model
from torch.nn.parallel import DistributedDataParallel
import time

def compute_average(epoch):
    return sum(epoch[1:])/(len(epoch)-1)

def init_process(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=size)


def train_cache(rank: int, world_size, config: RunConfig, indptr, indices, edge_id,
                train_idx, test_idx, valid_idx, feats, labels, partition_map, cached_ids):
    assert(config.is_valid())
    config.rank = rank
    print(config)
    if(world_size != 1):
        init_process(rank, world_size)
    feats_handle = pin_memory_inplace(feats)
    indptr_handle = pin_memory_inplace(indptr)
    indices_handle = pin_memory_inplace(indices)
    edge_id_handle = pin_memory_inplace(edge_id)
    labels_id_handle = pin_memory_inplace(labels)
    if rank == 0:
        graph = dgl.DGLGraph(('csc',(indptr, indices, edge_id)))
    # graph = dgl.hetero_from_shared_memory(config.graph_name).formats("csc")
    max_pool_size = 1
    num_redundant_layer = config.num_redundant_layers
    block_type = get_block_type("src_to_dst")

    step = min([(int(idx.shape[0] / config.batch_size) + 1) for idx in train_idx])
    print("Number of steps ", step)
    train_idx = train_idx[rank]
    # train_idx = torch.split(train_idx, train_idx.shape[0] // config.world_size)[rank]
    dataloader = init_groot_dataloader(
        config.rank, config.world_size, block_type, config.rank, config.fanouts,
        config.batch_size,  num_redundant_layer, max_pool_size,
        indptr, indices, feats, labels,
        train_idx, valid_idx, test_idx, partition_map
    )
    feat_size = feats.shape[1]

    if cached_ids != None:
        cache_id = cached_ids[rank].to(rank)
        # cache_id = get_cache_ids_by_sampling(config, graph, train_idx)
        init_groot_dataloader_cache(cache_id)

    # sample one epoch before profiling
    # model = get_dgl_model(config).to(0)
    layer = config.model_type
    num_layers = len(config.fanouts)

    n_hidden = config.hid_feat
    n_classes = int((torch.max(labels) + 1).item())

    model = get_distributed_model( layer, feat_size, num_layers, n_hidden, n_classes, num_redundant_layer,
                                   rank, world_size).to(rank)

    model = DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epoch = config.num_epoch + 1
    train_stream = torch.cuda.Stream()
    train_event = torch.cuda.Event()
    train_event.record(train_stream)

    for epoch in range(num_epoch):
        print("Start training ")
        randInt = torch.randperm(train_idx.shape[0], device = rank)
        shuffle_training_nodes(randInt)
        edges_per_epoch = 0
        for i in range(step):
            key = sample_batch_sync()
            blocks, batch_feat, batch_label = get_batch(key, layers = num_layers, \
                                                        n_redundant_layers =num_redundant_layer , mode = "SRC_TO_DEST")
            local_blocks, _, _ = blocks
            for block in local_blocks:
                edges_per_epoch += block.num_edges()
            torch.cuda.synchronize()
            train_event.wait(train_stream)
            batch_feat.requires_grad = True
            pred = model(blocks, batch_feat)
            pred = pred @ torch.ones(2,1, device = rank)
            torch.sum(batch_label * pred.flatten()).backward()

def test_bfs_flow():
    t1 = time.time()
    config = RunConfig()
    indptr, indices, edge_id, shared_graph, train_idx, test_idx, valid_idx, feat, \
        label, partition_map, cached_ids= load_dgl_dataset(config)
    t2 = time.time()
    print("total time to load data", t2 -t1)
    assert(config.is_valid())
    import gc
    gc.collect()
    if config.world_size == 1:
        train_cache(config.rank, config.world_size, config, indptr, indices, edge_id, train_idx, test_idx, valid_idx, feat, label, partition_map)
    else:
        mp.spawn(train_cache, args=(config.world_size, config, indptr, indices, edge_id, train_idx, test_idx, valid_idx, feat, label, partition_map, cached_ids), nprocs=config.world_size, daemon=True)# def groot_cache(config: RunConfig):


if __name__ == "__main__":
    test_bfs_flow()
