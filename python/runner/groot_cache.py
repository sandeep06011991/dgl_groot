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
        init_groot_dataloader_cache(cache_id.to(indptr.dtype))
    else:
        print("skipping cache")
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
    epoch_times = []
    epoch_timer = Timer()
    sampling_times = []
    sampling_timer = Timer()
    training_times = []
    training_timer = Timer()
    max_memory_used = []
    edges_computed = []
    for epoch in range(num_epoch):
        # if epoch == 0:
        #     print("pre-heating")
        # else:
        #     if epoch == 1:
        #         torch.cuda.synchronize()
        #         print(f"pre-heating takes {round(epoch_timer.passed(), 2)} sec")
        #         epoch_timer.reset()
        #     print(f"start epoch {epoch}")
        print("Start training ")
        randInt = torch.randperm(train_idx.shape[0], device = rank)
        shuffle_training_nodes(randInt)
        epoch_timer.reset()
        sampling_timer.reset()
        training_timer.reset()
        epoch_timer.start()
        edges_per_epoch = 0
        for i in range(step):
            sampling_timer.start()
            extract_feat_label_flag = False
            key = sample_batch_sync(extract_feat_label_flag)
            sampling_timer.stop()

            is_async = False
            extract_batch_feat_label(key, is_async)

            blocks, batch_feat, batch_label = get_batch(key, layers = num_layers, \
                                                        n_redundant_layers = config.num_redundant_layers , mode = "SRC_TO_DEST")
            local_blocks, _, _ = blocks
            for block in local_blocks:
                edges_per_epoch += block.num_edges()
            torch.cuda.synchronize()
            sampling_timer.stop()
            if config.sample_only:
                continue
            train_event.wait(train_stream)
            # wait for previous training to complete
            # with torch.cuda.stream(train_stream):
            training_timer.start()
            pred = model(blocks, batch_feat)
            torch.cuda.synchronize()
            loss = torch.nn.functional.cross_entropy(pred, batch_label)
            optimizer.zero_grad()
            training_timer.stop()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            sampling_timer.accumulate_async()
            training_timer.accumulate_async()
        max_memory_used.append(torch.cuda.memory_allocated())
        passed = epoch_timer.stop()
        epoch_times.append(epoch_timer.get_accumulated())
        training_times.append(training_timer.get_async_accumulated())
        sampling_times.append(sampling_timer.get_async_accumulated())
        edges_computed.append(edges_per_epoch)
    torch.cuda.synchronize()
    if config.test_acc  and rank == 0:
        use_uva = True
        graph.ndata["feat"] = feats
        graph.ndata["label"] = labels
        if use_uva:
            graph.pin_memory_()
            device = torch.device(rank)
        else:
            device = torch.device('cpu')
        model = model.to(device)
        test_idx = test_idx.to(device)

        test_dtype = torch.int64

        sampler = dgl.dataloading.NeighborSampler(\
                config.fanouts, prefetch_node_feats=['feat'], prefetch_labels=['label'])
        test_dataloader = dgl.dataloading.DataLoader(graph = graph.astype(test_dtype),
                                                indices = test_idx.to(test_dtype),
                                                graph_sampler = sampler,
                                                use_uva=use_uva, drop_last = True,
                                                batch_size=config.batch_size)

        acc = test_model_accuracy(config, model, test_dataloader)
        print("accuracy:", acc )
        print("epoch_time:",    avg_ignore_first(epoch_times))
        print("sampling_time", avg_ignore_first(sampling_times))
        print("training_time:", avg_ignore_first(training_times))
        max_cache_fraction  = min(1.0, int(((16 * 1024 ** 3) - max(max_memory_used)) / (feats.shape[1] * 4)))
        print("max_cache_fraction:", max_cache_fraction)
    print("edges_computed:", sum(edges_computed)/len(edges_computed))
    torch.distributed.barrier()
    print(f"Max memory used:{'{:.2f}'.format(max(max_memory_used) / (1024 ** 3))}GB")


def groot_cache(config: RunConfig):

    t1 = time.time()
    indptr, indices, edge_id, shared_graph, train_idx, test_idx, valid_idx, feat,\
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
