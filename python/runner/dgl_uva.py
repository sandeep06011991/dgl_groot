import torch, dgl
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from .util import *
from .dgl_model import get_dgl_model

def dgl_uva(config: RunConfig):
    # load dgl data graph
    dgl_dataset = load_dgl_dataset(config)    
    assert(config.is_valid())
    print(config)
    
    graph: dgl.DGLGraph = dgl_dataset.graph
    graph.pin_memory_()
    sampler = dgl.dataloading.NeighborSampler(config.fanouts, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    dataloader = dgl.dataloading.DataLoader(graph = graph, 
                                            indices = dgl_dataset.train_idx,
                                            graph_sampler = sampler, 
                                            use_uva=True,
                                            batch_size=config.batch_size)
    
    # sample one epoch before profiling
    model = get_dgl_model(config).to(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("pre-heating")
    step = 0
    for _, _, blocks in dataloader:
        if not config.sample_only:
            batch_feat = blocks[0].srcdata["feat"]
            batch_label = blocks[-1].dstdata["label"]
            batch_pred = model(blocks, batch_feat)
            batch_loss = F.cross_entropy(batch_pred, batch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if step == 0:
                print(f"{step=}")
                print(batch_feat)
                for block in blocks:
                    print(block)
            step += 1
    print("start training")
    timer = Timer()
    
    for num_epoch in range(config.num_epoch):
        print(f"start {num_epoch + 1} epoch")
        for _, _, blocks in dataloader:
            if not config.sample_only:
                batch_feat = blocks[0].srcdata["feat"]
                batch_label = blocks[-1].dstdata["label"]
                batch_pred = model(blocks, batch_feat)
                batch_loss = F.cross_entropy(batch_pred, batch_label)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

    passed = timer.passed()
    duration = passed / config.num_epoch
    
    print(f"duration={round(duration,2)} secs / epoch")