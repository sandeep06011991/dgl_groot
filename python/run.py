import dgl
from runner.util import *
from runner.dgl_uva import dgl_uva
from runner.dgl_gpu import dgl_gpu
from runner.groot_uva import groot_uva
from runner.groot_cache import groot_cache
from runner.groot_gpu import groot_gpu

if __name__ == "__main__":
    config = get_config()
    config.fanouts = [20,20,20]
    config.random_partition = False
    config.graph_name = "ogbn-arxiv"
    # config.system = "groot-cache"
    config.batch_size = 128
    # config.graph_name = "ogbn-products"
    config.world_size = 4
    config.cache_percentage = .5
    # config.graph_name = "ogbn-papers100M"
    config.test_acc = True
    config.model_type = "gat"
    config.num_epoch = 5
    if config.system == "dgl-uva":
        dgl_uva(config)
    elif config.system == "dgl-gpu":
        dgl_gpu(config)
    elif config.system == "groot-uva":
        groot_uva(config)
    elif config.system == "groot-cache":
        groot_cache(config)
    elif config.system == "groot-gpu":
        groot_gpu(config)
