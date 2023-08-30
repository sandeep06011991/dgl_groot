from runner.util import *
from runner.dgl_uva import dgl_uva
from runner.groot_uva import groot_uva
if __name__ == "__main__":
    config = get_config()
    config.fanouts = [15, 10, 5]
    
    if config.system == "dgl-uva":
        dgl_uva(config)
    elif config.system == "groot-uva":
        groot_uva(config)