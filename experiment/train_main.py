import os
from utils import get_partition_type, get_args, Config

if __name__ == "__main__":

    args = get_args()    
    graph_name = args.graph_name
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    node_weight = args.node_weight
    edge_weight = args.edge_weight
    bal = args.bal
    system = args.system
    sample_mode = args.sample_mode
    world_size = args.world_size
    model = args.model
    cache_size = args.cache_size
    hid_size = args.hid_size
    fanouts = args.fanouts.split(',')
    log_file=args.log_file 
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, f"logs/{log_file}")
    cfg = Config(graph_name=graph_name,
                    world_size=world_size,
                    num_partition=world_size,
                    num_epoch=num_epoch,
                    fanouts=fanouts,
                    batch_size=batch_size,
                    system=system,
                    model=model,
                    cache_size=cache_size,
                    hid_size=hid_size,
                    log_path=log_path,
                    data_dir=data_dir,
                    nvlink=args.nvlink,
                    partition_type=get_partition_type(node_weight, edge_weight, bal),
                    sample_mode=sample_mode)
    cfg.num_redundant_layer = args.num_redundant_layer
    cfg.cache_frac = args.cache_frac
    cfg.random_feat = args.random_feat
    assert 0 <= cfg.num_redundant_layer <= len(fanouts), \
        f"num_redundant_layer (k={cfg.num_redundant_layer}) must be in [0, {len(fanouts)}]"
    if cfg.system == "split":
        from nodepred.trainer import bench_split
        bench_split(cfg)
    elif cfg.system == "gsplit_pipepar":
        from nodepred.trainer import bench_pipeline_split
        bench_pipeline_split(cfg)
    elif cfg.system == "dgl":
        from nodepred.trainer import bench_dgl_batch
        bench_dgl_batch([cfg])
    elif cfg.system == "quiver":
        from nodepred.quiver_trainer import bench_quiver_batch
        bench_quiver_batch([cfg])
    elif cfg.system == "dist_cache":
        from nodepred.quiver_trainer import bench_quiver_batch
        bench_quiver_batch([cfg])
    elif cfg.system == "p3":
        from nodepred.p3_trainer import bench_p3_batch
        bench_p3_batch([cfg])

