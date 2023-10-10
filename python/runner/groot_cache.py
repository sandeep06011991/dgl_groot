import torch, dgl
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from .util import *
from .dgl_model import get_dgl_model
from dgl.groot import *
from dgl.utils import pin_memory_inplace
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from dgl.groot.models import get_distributed_model
from torch.nn.parallel import DistributedDataParallel
import time

def init_process(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=size)

def train_cache(rank: int, world_size, config: RunConfig, indptr, indices, edge_id, train_idx, test_idx, valid_idx, feats, labels, partition_map):
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
    graph = dgl.DGLGraph(('csc',(indptr, indices, edge_id)))
    # graph = dgl.hetero_from_shared_memory(config.graph_name).formats("csc")
    max_pool_size = 1
    num_redundant_layer = config.num_redundant_layers
    block_type = get_block_type("src_to_dst")
    print("Start groot dataloader")
    train_idx = torch.split(train_idx, train_idx.shape[0] // config.world_size)[rank]
    dataloader = init_groot_dataloader(
        config.rank, config.world_size, block_type, config.rank, config.fanouts,
        config.batch_size,  num_redundant_layer, max_pool_size,
        indptr, indices, feats, labels,
        train_idx, valid_idx, test_idx, partition_map
    )
    feat_size = feats.shape[1]

    print("Get the cache ")
    cache_id = get_cache_ids_by_sampling(config, graph, train_idx)
    init_groot_dataloader_cache(cache_id)

    # sample one epoch before profiling
    # model = get_dgl_model(config).to(0)
    print("Hardcoding ")
    model_type = "GAT"
    layer = "GAT"
    num_layers = len(config.fanouts)

    n_hidden = 256
    if config.graph_name == "test-data":
        model_type = "test"
        n_classes = 1
    else:
        n_classes = int((torch.max(labels) + 1).item())

    num_redundant = num_redundant_layer
    model = get_distributed_model(model_type, layer, feat_size, num_layers, n_hidden, n_classes, num_redundant,
                                  rank, world_size).to(rank)
    if world_size != 1:
        pass
    if model_type != "test":
        model = DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = int(train_idx.shape[0] / config.batch_size) + 1
    num_epoch = config.num_epoch + 1
    train_stream = torch.cuda.Stream()
    train_event = torch.cuda.Event()
    train_event.record(train_stream)
    timer = Timer()
    for epoch in range(num_epoch +1 ):
        if epoch == 0:
            print("pre-heating")
        else:
            if epoch == 1:
                torch.cuda.synchronize()
                print(f"pre-heating takes {round(timer.passed(), 2)} sec")
                timer.reset()
            print(f"start epoch {epoch}")
        randInt = torch.randperm(train_idx.shape[0], device = rank)
        shuffle_training_nodes(randInt)
        for i in range(step):
            key = sample_batch_sync()
            blocks, batch_feat, batch_label = get_batch(key, layers = num_layers, \
                        n_redundant_layers =num_redundant_layer , mode = "SRC_TO_DEST")

            if config.sample_only:
                continue
            train_event.wait(train_stream)
            # wait for previous training to complete
            # with torch.cuda.stream(train_stream):
            if config.graph_name == "test-data":
                batch_feat.requires_grad = True
                pred = model(blocks, batch_feat)
                pred = pred @ torch.ones(2,1, device = rank)
                torch.sum(batch_label * pred.flatten()).backward()
            else:
                pred = model(blocks, batch_feat)
            torch.cuda.synchronize()
            if config.graph_name == "test-data":
                break
            loss = torch.nn.functional.cross_entropy(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    passed = timer.passed()
    duration = passed / config.num_epoch
    print(f"{config.rank} duration={round(duration,2)} secs / epoch")
    if config.test_acc:
        graph.ndata["feat"] = feats
        graph.ndata["label"] = labels
        graph.pin_memory_()
        sampler = dgl.dataloading.NeighborSampler(config.fanouts, prefetch_node_feats=['feat'], prefetch_labels=['label'])
        test_dataloader = dgl.dataloading.DataLoader(graph = graph,
                                                indices = test_idx,
                                                graph_sampler = sampler,
                                                use_uva=True,
                                                batch_size=config.batch_size)

        acc = test_model_accuracy(config, model, test_dataloader)
        print("Accuracy ", acc )


def groot_cache(config: RunConfig):
    print("start data load process")
    t1 = time.time()
    indptr, indices, edge_id, shared_graph, train_idx, test_idx, valid_idx, feat, label, partition_map = load_dgl_dataset(config)
    t2 = time.time()
    print("total time to load data", t2 -t1)
    assert(config.is_valid())
    import gc
    gc.collect()

    if config.world_size == 1:
        train_cache(config.rank, config.world_size, config, indptr, indices, edge_id, train_idx, test_idx, valid_idx, feat, label, partition_map)
    else:
        mp.spawn(train_cache, args=(config.world_size, config, indptr, indices, edge_id, train_idx, test_idx, valid_idx, feat, label, partition_map), nprocs=config.world_size, daemon=True)# def groot_cache(config: RunConfig):