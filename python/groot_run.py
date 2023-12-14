from exp.util import  *
from exp.groot_trainer import *


def get_data_dir(graph_name):
    if os.environ['MACHINE_NAME'] == 'jupyter':
        if "com" in graph_name:
            return "/data/sandeep/groot_data/snap/"
        if "ogbn" in graph_name:
            return "/data/sandeep/groot_data/ogbn-processed"

    if os.environ['MACHINE_NAME'] == "p3.8xlarge" or os.environ['MACHINE_NAME'] == 'p3.16xlarge':
        if "com" in graph_name:
            return "/data/snap/"
        if "ogbn" in graph_name:
            return "/data/ogbn/processed/"
    else:
        if "com" in graph_name:
            return "/ssd/snap/"
        if "ogbn" in graph_name:
            return "/ssd/ogbn/processed/"

if __name__ == "__main__":
    data_dir = get_data_dir("ogbn")
    log_path = "./log/dummy.csv"
    config = get_config( log_path, data_dir)
    in_dir = os.path.join(config.data_dir, config.graph_name)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
    test_acc = True
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    indptr, indices, edges = load_graph(in_dir, is32=True, wsloop=True)
    feat, label, num_label = load_feat_label(in_dir)
    print(config.partition_type)
    if config.partition_type == "random":
        partition_map = torch.randint(0, config.world_size, (graph.num_nodes(),))
    else:
        partition_map = get_metis_partition(in_dir, config, graph)
    train_idx_list = []
    for p in range(config.world_size):
        train_idx_list.append(train_idx[partition_map[train_idx] == p])
    config.cache_percentage = config.cache_size
    config.cache_rate = config.cache_percentage
    cached_ids = get_cache_ids_by_sampling(config, graph, train_idx_list, partition_map)
    spawn(train_ddp, args=(config, test_acc, graph, feat, label, \
                           num_label, train_idx_list, valid_idx, test_idx, \
                           indptr, indices, edges,  partition_map, cached_ids), \
          nprocs=config.world_size, daemon=True, join= True)