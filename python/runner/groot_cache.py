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

def init_process(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=size)

def train_cache(rank: int, world_size, config: RunConfig,  dgl_dataset: DGLDataset):
    assert(config.is_valid())
    config.rank = rank
    print(config)

    if(world_size != 1):
        init_process(rank, world_size)
    
    graph: dgl.DGLGraph = dgl_dataset.graph
    graph = graph.add_self_loop()
    indptr, indices, edge_id = graph.adj_tensors('csc')
    feats = graph.ndata.pop("feat")
    labels = graph.ndata.pop("label")
    
    feats_handle = pin_memory_inplace(feats)
    indptr_handle = pin_memory_inplace(indptr)
    indices_handle = pin_memory_inplace(indices)
    edge_id_handle = pin_memory_inplace(edge_id)
    labels_id_handle = pin_memory_inplace(labels)

    max_pool_size = 1
    num_redundant_layer = len(config.fanouts) - 1
    block_type = get_block_type("src_to_dst")
    partition_map = None
    dgl_dataset.train_idx = torch.split(dgl_dataset.train_idx, dgl_dataset.train_idx.shape[0] // config.world_size)[rank]
    dataloader = init_groot_dataloader(
        config.rank, config.world_size, block_type, config.rank, config.fanouts,
        config.batch_size,num_redundant_layer, max_pool_size, 
        indptr, indices, feats, labels,
        dgl_dataset.train_idx, dgl_dataset.valid_idx, dgl_dataset.test_idx, partition_map
    )
    feat_size = feats.shape[1]
    cache_id = get_cache_ids_by_sampling(config, graph, dgl_dataset.train_idx)
    init_groot_dataloader_cache(cache_id)
    
    # sample one epoch before profiling
    # model = get_dgl_model(config).to(0)
    print("Hardcoding ")
    mode = "SRC_TO_DEST"
    layer = "GAT"
    num_layers = 3
    n_hidden = 256
    n_classes = torch.max(labels) + 1
    num_redundant = 2
    model = get_distributed_model(mode, layer, feat_size, num_layers, n_hidden, n_classes, num_redundant,
                                  rank, world_size).to(rank)
    model = DistributedDataParallel(model)

# model = get_distributed_model(config).to(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = int(dgl_dataset.train_idx.shape[0] / config.batch_size)
    feat_size_in_bytes = 0
    key = -1
    num_epoch = config.num_epoch + 1
    train_stream = torch.cuda.Stream()
    train_event = torch.cuda.Event()
    train_event.record(train_stream)
    duration_in_s = 0
    timer = Timer()
    for epoch in range(num_epoch):
        if epoch == 0:
            print("pre-heating")
        else:
            if epoch == 1:
                torch.cuda.synchronize()    
                print(f"pre-heating takes {round(timer.passed(), 2)} sec")
                timer.reset()
            print(f"start epoch {epoch}")
                
        for i in range(step):
            # if key == -1:
            #   pass
            key = sample_batch_sync()
            blocks, batch_feat, batch_label = get_batch(key, layers = num_layers, \
                        n_redundant_layers =num_redundant_layer , mode = "SRC_TO_DEST")

            # prefetch next mini-batch
            # key = sample_batch_async()
            # blocks, batch_feat, batch_label = get_batch(sample_batch_sync())            
            if config.sample_only:
                continue
            
            train_event.wait(train_stream)            
            # wait for previous training to complete
            with torch.cuda.stream(train_stream):
                pred = model(blocks, batch_feat)
                loss = torch.nn.functional.cross_entropy(pred, batch_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # train_event.record(train_stream)

            train_stream.synchronize()
            if epoch > 0:
                num_feat, feat_width = batch_feat.shape
                feat_size_in_bytes += num_feat * feat_width * 4
                
            if epoch == 0 and i == 0:
                print(batch_feat)
                print(batch_label)
                for block in blocks:
                    print(block)
                    
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
                                                indices = dgl_dataset.test_idx,
                                                graph_sampler = sampler, 
                                                use_uva=True,
                                                batch_size=config.batch_size)
        
        acc = test_model_accuracy(config, model, test_dataloader)
        print("Accuracy ", acc )


def groot_cache(config: RunConfig):
    dgl_dataset: DGLDataset = load_dgl_dataset(config)    
    assert(config.is_valid())
    print(config.world_size)
    if config.world_size == 1:
        train_cache(config.rank, config.world_size, config, dgl_dataset)
    else:
        mp.spawn(train_cache, args=(config.world_size, config, dgl_dataset), nprocs=config.world_size, daemon=True)# def groot_cache(config: RunConfig):

#     # load dgl data graph
#     dgl_dataset: DGLDataset = load_dgl_dataset(config)    
#     assert(config.is_valid())
#     print(config)
    
#     graph: dgl.DGLGraph = dgl_dataset.graph    
#     indptr, indices, edge_id = graph.adj_tensors('csc')
#     feats = graph.ndata.pop("feat")
#     labels = graph.ndata.pop("label")
    
#     feats_handle = pin_memory_inplace(feats)
#     indptr_handle = pin_memory_inplace(indptr)
#     indices_handle = pin_memory_inplace(indices)
#     edge_id_handle = pin_memory_inplace(edge_id)
#     labels_id_handle = pin_memory_inplace(labels)


#     max_pool_size = 2   
#     num_redundant_layer = len(config.fanouts) - 1
#     block_type = get_block_type("src_to_dst")
#     partition_map = None
#     dataloader = init_groot_dataloader(
#         config.rank, config.world_size, block_type, config.rank, config.fanouts,
#         config.batch_size,num_redundant_layer, max_pool_size, 
#         indptr, indices, feats, labels,
#         dgl_dataset.train_idx, dgl_dataset.valid_idx, dgl_dataset.test_idx, partition_map
#     )
    
#     cache_id = get_cache_ids_by_sampling(config, graph, dgl_dataset.train_idx)
#     init_groot_dataloader_cache(cache_id)
    
#     # sample one epoch before profiling
#     model = get_dgl_model(config).to(0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
#     step = int(dgl_dataset.train_idx.shape[0] / config.batch_size)
#     feat_size_in_bytes = 0
#     key = -1
#     num_epoch = config.num_epoch + 1
#     train_stream = torch.cuda.Stream()
#     train_event = torch.cuda.Event()
#     train_event.record(train_stream)
#     duration_in_s = 0
#     timer = Timer()
#     for epoch in range(num_epoch):
#         if epoch == 0:
#             print("pre-heating")
#         else:
#             if epoch == 1:
#                 torch.cuda.synchronize()    
#                 print(f"pre-heating takes {round(timer.passed(), 2)} sec")
#                 timer.reset()
#             print(f"start epoch {epoch}")
                
#         for i in range(step):
#             if key == -1:
#                 key = sample_batch_sync()
#             blocks, batch_feat, batch_label = get_batch(key)
#             # prefetch next mini-batch
#             key = sample_batch_async()
#             # blocks, batch_feat, batch_label = get_batch(sample_batch_sync())            
#             if config.sample_only:
#                 continue
            
#             train_event.wait(train_stream)            
#             # wait for previous training to complete
#             with torch.cuda.stream(train_stream):
#                 pred = model(blocks, batch_feat)
#                 loss = torch.nn.functional.cross_entropy(pred, batch_label)
#                 optimizer.zero_grad()            
#                 loss.backward()
#                 optimizer.step()
#                 train_event.record(train_stream)

#             if epoch > 0:
#                 num_feat, feat_width = batch_feat.shape
#                 feat_size_in_bytes += num_feat * feat_width * 4
                
#             if epoch == 0 and i == 0:
#                 print(batch_feat)
#                 print(batch_label)
#                 for block in blocks:
#                     print(block)
                    
#     torch.cuda.synchronize()        
#     passed = timer.passed()
#     duration = passed / config.num_epoch
    
#     print(f"{config.rank} duration={round(duration,2)} secs / epoch")
#     if config.test_acc:
#         graph.ndata["feat"] = feats
#         graph.ndata["label"] = labels
#         graph.pin_memory_()
#         sampler = dgl.dataloading.NeighborSampler(config.fanouts, prefetch_node_feats=['feat'], prefetch_labels=['label'])
#         test_dataloader = dgl.dataloading.DataLoader(graph = graph, 
#                                                 indices = dgl_dataset.test_idx,
#                                                 graph_sampler = sampler, 
#                                                 use_uva=True,
#                                                 batch_size=config.batch_size)
        
#         acc = test_model_accuracy(config, model, test_dataloader)