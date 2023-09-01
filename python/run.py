from runner.util import *
from runner.dgl_uva import dgl_uva
from runner.dgl_gpu import dgl_gpu
from runner.groot_uva import groot_uva
from runner.groot_cache import groot_cache

if __name__ == "__main__":
    config = get_config()
    config.fanouts = [15, 10, 5]
    
    if config.system == "dgl-uva":
        dgl_uva(config)
    elif config.system == "dgl-gpu":
        dgl_gpu(config)
    elif config.system == "groot-uva":
        groot_uva(config)
    elif config.system == "groot-cache":
        groot_cache(config)