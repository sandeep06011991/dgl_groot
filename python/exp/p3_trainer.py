import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.multiprocessing.spawn import ProcessContext, _wrap

from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows
import multiprocessing as MP

from .p3_model import *
from .util import *
import time

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
# Output: feat
def get_p3_local_feat(rank: int, world_size:int, feat: torch.Tensor, padding=True) -> torch.Tensor:
    org_feat_width = feat.shape[1]
    if padding and org_feat_width % world_size != 0:
        step = int(org_feat_width / world_size)
        pad = world_size - org_feat_width + step * world_size
        padded_width = org_feat_width + pad
        assert(padded_width % world_size == 0)
        step = int(padded_width / world_size)
        start_idx = rank * step
        end_idx = start_idx + step
        local_feat = None
        if rank == world_size - 1:
            # padding is required for P3 to work correctly
            local_feat = feat[:, start_idx : org_feat_width]
            zeros = torch.zeros((local_feat.shape[0], pad), dtype=local_feat.dtype)
            local_feat = torch.concatenate([local_feat, zeros], dim=1)
        else:
            local_feat = feat[:, start_idx : end_idx]
        return local_feat
    else:
        step = int(feat.shape[1] / world_size)
        start_idx = rank * step
        end_idx = min(start_idx + step, feat.shape[1])
        if rank == world_size - 1:
            end_idx = feat.shape[1]
        local_feat = feat[:, start_idx : end_idx]
        return local_feat
    
    
def bench_p3_batch(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    print("Start data loading")
    t1 = time.time()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
    graph.create_formats_()
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    feat, label, num_label = load_feat_label(in_dir)
    feats = [None] * config.world_size
    in_feat = -1
    for i in range(config.world_size):
        feats[i] = get_p3_local_feat(i, config.world_size, feat, padding=True).clone()
        if i == 0:
            in_feat = feats[i].shape[1]
        else:
            assert(in_feat == feats[i].shape[1])
    del feat
    
    print("Data loading total time", time.time() - t1)
    for config in configs:
        config.num_classes = num_label
        config.in_feat = in_feat
        if config.graph_name  == "com-orkut":
            config.system = "p3-gpu"
            config.cache_size = 0
        if config.graph_name == "ogbn-products":
            config.system = "p3-gpu"
            config.cache_size = 1
        if config.graph_name == "ogbn-papers100M" or config.graph_name == "com-friendster":
            config.system = "p3-uva"
            config.cache_size = 0
        try:
            spawn(train_ddp, args=(config, test_acc, graph, feats, label, num_label, train_idx, valid_idx, test_idx), nprocs=config.world_size, daemon=True)
            # nprocs = config.world_size
            # mp = MP.get_context("spawn")
            # error_queues = []
            # processes = []
            # for i in range(nprocs):
            #     error_queue = mp.SimpleQueue()
            #     args = (config, test_acc, graph, feats[i], label, num_label, train_idx, valid_idx, test_idx)
            #     process = mp.Process(
            #         target=_wrap,
            #         args=(train_ddp, i, args, error_queue),
            #         daemon=True,
            #     )
            #     process.start()
            #     error_queues.append(error_queue)
            #     processes.append(process)

            # context = ProcessContext(processes, error_queues)
            # # Loop on join until it returns True or raises an exception.
            # while not context.join():
            #     pass
        except Exception as e:
            print(e)
            if "out of memory"in str(e):
                write_to_csv(config.log_path, [config], [oom_profiler()])
            else:
                write_to_csv(config.log_path, [config], [empty_profiler()])
                with open(f"exceptions/{config.get_file_name()}",'w') as fp:
                    fp.write(str(e))
        gc.collect()
        torch.cuda.empty_cache()
            
def train_ddp(rank: int, config: Config, test_acc: bool,
              graph: dgl.DGLGraph, feats: list[torch.Tensor], label: torch.Tensor, num_label: int, 
              train_idx: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    start_time = time.time()
    graph_sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    dataloader, graph = get_dgl_sampler(graph=graph, train_idx=train_idx,
                                        graph_samler=graph_sampler, system=config.system, batch_size=config.batch_size, use_dpp=True)
    assert config.cache_size in [0, 1]
    feat = feats[rank] # load local feature
    if config.cache_size == 0:
        feat_nd  = pin_memory_inplace(feat)
        label_nd = pin_memory_inplace(label)
    if config.cache_size == 1:
        feat = feat.to(device)
        label = label.to(device)
        
    local_model, global_model = get_p3_model(config)
    print(f"creating ddp on device: {device}")
    global_model = DDP(global_model, device_ids=[rank], output_device=rank)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    print(f"creating buffers on device: {device}")
    edge_size_lst: list = [(0, 0, 0, 0)] * config.world_size #(rank, num_edges, num_dst_nodes, num_src_nodes)
    est_node_size = config.batch_size * 20
    local_feat_width = feat.shape[1]
    input_node_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
    input_feat_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
    src_edge_buffer_lst: list[torch.Tensor] = [] # storing src nodes
    dst_edge_buffer_lst: list[torch.Tensor] = [] # storing dst nodes 
    global_grad_lst: list[torch.Tensor] = [] # storing feature data gathered for other gpus
    local_hid_buffer_lst: list[torch.Tensor] = [None] * config.world_size # storing feature data gathered from other gpus
    for idx in range(config.world_size):    
        nid_dtype = train_idx.dtype   
        input_node_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        src_edge_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        dst_edge_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        global_grad_lst.append(torch.zeros([est_node_size, config.hid_size], dtype=torch.float32, device=device))
        input_feat_buffer_lst.append(torch.zeros([est_node_size, config.hid_size], dtype=torch.float32, device=device))

    shuffle = P3Shuffle.apply
    print(f"pre-heating model on device: {device}")
    dist.barrier(device_ids=[i for i in range(config.world_size)])
    for input_nodes, output_nodes, blocks in dataloader:
        top_block: dgl.DGLGraph = blocks[0]
        src, dst = top_block.adj_tensors("coo")
        edge_size_lst[rank] = (rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes())
        dist.all_gather(edge_size_lst, edge_size_lst[rank])
        
        for rank, edge_size, src_node_size, dst_node_size in edge_size_lst:
            src_edge_buffer_lst[rank].resize_(edge_size)
            dst_edge_buffer_lst[rank].resize_(edge_size)
            input_node_buffer_lst[rank].resize_(src_node_size)
            
        batch_feat = None
        handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
        handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
        handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)

        batch_label = None
        if config.cache_size == 0:
            batch_label = gather_pinned_tensor_rows(label, output_nodes)
        if config.cache_size == 1:
            batch_label = label[output_nodes]
            
        handle1.wait()
        for rank, _input_nodes in enumerate(input_node_buffer_lst):
            if config.cache_size == 1:
                input_feat_buffer_lst[rank] = feat[_input_nodes]
            elif config.cache_size == 0:
                input_feat_buffer_lst[rank] = gather_pinned_tensor_rows(feat, _input_nodes)
            else: # 'caching'
                print("Current p3 implementation does not support caching")
                exit(-1)
                
        handle2.wait()
        handle3.wait()
        # print(f"{rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in input_feat_buffer_lst]} start computing first hidden layer")
        torch.cuda.synchronize()
        block = None
        for r in range(config.world_size):
            input_nodes = input_node_buffer_lst[r]
            input_feats = input_feat_buffer_lst[r]
            if r == rank:
                block = top_block
            else:
                src = src_edge_buffer_lst[r]
                dst = dst_edge_buffer_lst[r]
                src_node_size = edge_size_lst[r][2]
                dst_node_size = edge_size_lst[r][3]
                block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
                                    
            local_hid_buffer_lst[r] = local_model(block, input_feats)
            global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
    
    
        agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                          local_hid_buffer_lst[rank], 
                                          local_hid_buffer_lst, 
                                          global_grad_lst)
        # 6. Compute forward pass locally
        batch_pred = global_model(blocks[1:], agg_hid)
        batch_loss = F.cross_entropy(batch_pred, batch_label)
        torch.cuda.synchronize()
        
        # backward
        global_optimizer.zero_grad()
        local_optimizer.zero_grad()
        batch_loss.backward()
        
        global_optimizer.step()
        for idx, global_grad in enumerate(global_grad_lst):
            if idx != rank:
                local_optimizer.zero_grad()
                local_hid_buffer_lst[idx].backward(global_grad)
                local_optimizer.step()
        torch.cuda.synchronize()
    
    dist.barrier()

    print(f"training model on device: {device}")        
    timer = Timer()
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    edges_computed = []
    for epoch in range(config.num_epoch):
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"start epoch {epoch}")
        edges_computed_epoch = 0
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            sampling_timer.end()
            
            # 2. feature extraction and shuffling
            feat_timer = CudaTimer()

            top_block: dgl.DGLGraph = blocks[0]
            src, dst = top_block.adj_tensors("coo")
            edge_size_lst[rank] = (rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes())
            dist.all_gather(edge_size_lst, edge_size_lst[rank])
            
            for rank, edge_size, src_node_size, dst_node_size in edge_size_lst:
                src_edge_buffer_lst[rank].resize_(edge_size)
                dst_edge_buffer_lst[rank].resize_(edge_size)
                input_node_buffer_lst[rank].resize_(src_node_size)
                
            batch_feat = None
            handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
            handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
            handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)

            batch_label = None
            if config.cache_size == 0:
                batch_label = gather_pinned_tensor_rows(label, output_nodes)
            if config.cache_size == 1:
                batch_label = label[output_nodes]
                
            handle1.wait()
            for rank, _input_nodes in enumerate(input_node_buffer_lst):
                if config.cache_size == 1:
                    input_feat_buffer_lst[rank] = feat[_input_nodes]
                elif config.cache_size == 0:
                    input_feat_buffer_lst[rank] = gather_pinned_tensor_rows(feat, _input_nodes)
                else: # 'caching'
                    print("Current p3 implementation does not support caching")
                    exit(-1)
                    
            handle2.wait()
            handle3.wait()
            # print(f"{rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in input_feat_buffer_lst]} start computing first hidden layer")
            torch.cuda.synchronize()
            feat_timer.end()            

            # 3. compute hid tensor for all ranks
            forward_timer = CudaTimer()
            block = None
            for r in range(config.world_size):
                input_nodes = input_node_buffer_lst[r]
                input_feats = input_feat_buffer_lst[r]
                if r == rank:
                    block = top_block
                else:
                    src = src_edge_buffer_lst[r]
                    dst = dst_edge_buffer_lst[r]
                    src_node_size = edge_size_lst[r][2]
                    dst_node_size = edge_size_lst[r][3]
                    block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
                                        
                local_hid_buffer_lst[r] = local_model(block, input_feats)
                global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
        
        
            agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                            local_hid_buffer_lst[rank], 
                                            local_hid_buffer_lst, 
                                            global_grad_lst)
            # 6. Compute forward pass locally
            batch_pred = global_model(blocks[1:], agg_hid)
            batch_loss = F.cross_entropy(batch_pred, batch_label)
            torch.cuda.synchronize()
            forward_timer.end()

            # backward
            backward_timer = CudaTimer()
            global_optimizer.zero_grad()
            local_optimizer.zero_grad()
            batch_loss.backward()
            
            global_optimizer.step()
            for idx, global_grad in enumerate(global_grad_lst):
                if idx != rank:
                    local_optimizer.zero_grad()
                    local_hid_buffer_lst[idx].backward(global_grad)
                    local_optimizer.step()
                    
            torch.cuda.synchronize()
            backward_timer.end()
            for block in blocks:
                edges_computed_epoch += block.num_edges()
            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)
            sampling_timer = CudaTimer()
        edges_computed.append(edges_computed_epoch)

    # for epoch in range(config.num_epoch):
    #     if rank == 0 and (epoch + 1) % 5 == 0:
    #         print(f"start epoch {epoch}")
    #     edges_computed_epoch = 0
    #     sampling_timer = CudaTimer()
    #     for input_nodes, output_nodes, blocks in dataloader:
    #         sampling_timer.end()
            
    #         feat_timer = CudaTimer()
    #         batch_feat = None
    #         batch_label = None
    #         if config.cache_size == 0:
    #             batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
    #             batch_label = gather_pinned_tensor_rows(label, output_nodes)
    #         elif config.cache_size == 1:
    #             batch_feat = feat[input_nodes]
    #             batch_label = label[output_nodes]

    #         feat_timer.end()            
            
    #         forward_timer = CudaTimer()
    #         for block in blocks:
    #             edges_computed_epoch += block.num_edges()
    #         batch_pred = model(blocks, batch_feat)
    #         batch_loss = F.cross_entropy(batch_pred, batch_label)
    #         forward_timer.end()
            
    #         backward_timer = CudaTimer()
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()
    #         backward_timer.end()
            

    #         sampling_timers.append(sampling_timer)
    #         feature_timers.append(feat_timer)
    #         forward_timers.append(forward_timer)
    #         backward_timers.append(backward_timer)

    #         sampling_timer = CudaTimer()
    #     edges_computed.append(edges_computed_epoch)
    
    torch.cuda.synchronize()
    duration = timer.duration()
    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers)
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)
    profiler = Profiler(duration=duration, sampling_time=sampling_time, feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=0)
    profile_edge_skew(edges_computed, profiler, rank, dist)
    if rank == 0:
        print(f"train for {config.num_epoch} epochs in {duration}s")
        print(f"finished experiment {config} in {e2eTimer.duration()}s")
        if not test_acc:
            write_to_csv(config.log_path, [config], [profiler])
    
    dist.barrier()
    
    if test_acc:
        print(f"testing model accuracy on {device}")
        dataloader, graph = get_dgl_sampler(graph=graph, train_idx=test_idx, graph_samler=graph_sampler, system=config.system, batch_size=config.batch_size, use_dpp=True)    
        local_model.eval()
        global_model.eval()
        ys = []
        y_hats = []
        
        # for input_nodes, output_nodes, blocks in dataloader:
        #     with torch.no_grad():
        #         batch_feat = None
        #         batch_label = None
        #         if config.cache_size == 0:
        #             batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
        #             batch_label = gather_pinned_tensor_rows(label, output_nodes)
        #         else:
        #             batch_feat = feat[input_nodes]
        #             batch_label = label[output_nodes]
        #         # else:
        #         #     batch_feat = feat[input_nodes].to(device)
        #         #     batch_label = label[output_nodes].to(device)
        #         #     blocks = [block.to(device) for block in blocks]
        #         ys.append(batch_label)
        #         batch_pred = model(blocks, batch_feat)
        #         y_hats.append(batch_pred)
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                top_block: dgl.DGLGraph = blocks[0]
                src, dst = top_block.adj_tensors("coo")
                edge_size_lst[rank] = (rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes())
                dist.all_gather(edge_size_lst, edge_size_lst[rank])
                
                for rank, edge_size, src_node_size, dst_node_size in edge_size_lst:
                    src_edge_buffer_lst[rank].resize_(edge_size)
                    dst_edge_buffer_lst[rank].resize_(edge_size)
                    input_node_buffer_lst[rank].resize_(src_node_size)
                    
                batch_feat = None
                handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
                handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
                handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)

                batch_label = None
                if config.cache_size == 0:
                    batch_label = gather_pinned_tensor_rows(label, output_nodes)
                if config.cache_size == 1:
                    batch_label = label[output_nodes]
                    
                handle1.wait()
                for rank, _input_nodes in enumerate(input_node_buffer_lst):
                    if config.cache_size == 1:
                        input_feat_buffer_lst[rank] = feat[_input_nodes]
                    elif config.cache_size == 0:
                        input_feat_buffer_lst[rank] = gather_pinned_tensor_rows(feat, _input_nodes)
                    else: # 'caching'
                        print("Current p3 implementation does not support caching")
                        exit(-1)
                        
                handle2.wait()
                handle3.wait()
                # print(f"{rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in input_feat_buffer_lst]} start computing first hidden layer")
                torch.cuda.synchronize()
                block = None
                for r in range(config.world_size):
                    input_nodes = input_node_buffer_lst[r]
                    input_feats = input_feat_buffer_lst[r]
                    if r == rank:
                        block = top_block
                    else:
                        src = src_edge_buffer_lst[r]
                        dst = dst_edge_buffer_lst[r]
                        src_node_size = edge_size_lst[r][2]
                        dst_node_size = edge_size_lst[r][3]
                        block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
                                            
                    local_hid_buffer_lst[r] = local_model(block, input_feats)
                    global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
            
            
                agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                                local_hid_buffer_lst[rank], 
                                                local_hid_buffer_lst, 
                                                global_grad_lst)
                # 6. Compute forward pass locally
                ys.append(batch_label)
                batch_pred = global_model(blocks[1:], agg_hid)
                y_hats.append(batch_pred)
                # torch.cuda.synchronize()
                
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc = round(acc.item() * 100 / config.world_size, 2)
        profiler.test_acc = acc  
        if rank == 0:
            profiler.run_time = time.time() - start_time
            print(f"test accuracy={acc}%")
            write_to_csv(config.log_path, [config], [profiler])
            
            
    ddp_exit()
