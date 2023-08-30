from runner.util import *
from runner.dgl_uva import *

if __name__ == "__main__":
    config = get_config()
    config.fanouts = [15, 10, 5]
    dgl_uva(config)