"""record_timeout.py — write a single "timeout" row to the sweep CSV for a cell
that exceeded the per-run wall-clock cap (sweep.sh RUN_TIMEOUT). Takes the SAME
args as train_main.py so the Config/schema matches exactly; no data load, no
training. Used by sweep.sh when `timeout` kills a cell (rc 124/137)."""
import os
from utils import get_partition_type, get_args, Config
from utils.profiler import timeout_profiler
from utils.utils import write_to_csv

if __name__ == "__main__":
    args = get_args()
    fanouts = [int(x) for x in args.fanouts.split(',')]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, f"logs/{args.log_file}")
    cfg = Config(graph_name=args.graph_name,
                 world_size=args.world_size,
                 num_partition=args.world_size,
                 num_epoch=args.num_epoch,
                 fanouts=fanouts,
                 batch_size=args.batch_size,
                 system=args.system,
                 model=args.model,
                 cache_size=args.cache_size,
                 hid_size=args.hid_size,
                 log_path=log_path,
                 data_dir=args.data_dir,
                 nvlink=args.nvlink,
                 partition_type=get_partition_type(args.node_weight, args.edge_weight, args.bal),
                 sample_mode=args.sample_mode)
    cfg.num_redundant_layer = args.num_redundant_layer
    cfg.cache_frac = args.cache_frac
    cfg.random_feat = args.random_feat
    # in_feat is unknown without a data load; leave the -1 default.
    prof = timeout_profiler()
    write_to_csv(log_path, [cfg], [prof])
    print(f"recorded TIMEOUT row -> {log_path} for {cfg.get_file_name()} hid={cfg.hid_size}")
