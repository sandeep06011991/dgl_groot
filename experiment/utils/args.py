import argparse

def get_args():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--data_dir', required=True, type=str, help="Input graph directory")
    parser.add_argument('--batch_size', default=1024, type=int, help='Global batch size (default: 1024)')
    parser.add_argument('--num_epoch', default=2, type=int, help='Number of epochs to be sampled (default 2)')
    parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster","arxiv"])
    parser.add_argument('--system', default="split", type=str, help="System", choices=["dgl", "split","quiver","p3"])
    parser.add_argument('--model', default="sage", type=str, help="Model type", choices=["sage", "gat"])
    parser.add_argument('--hid_size', default=256, type=int, help="Model hidden dimension")
    parser.add_argument('--cache_size', default="0GB", type=str, help="Feature data cache size")
    parser.add_argument('--sample_mode', default="uva", type=str, help="Sample mode", choices=["uva", "gpu"])
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--num_partition', default=4, type=int, help='Number of partitions')
    parser.add_argument('--node_weight', default="dst", type=str, help="Node weight configuraion", choices=["uniform", "degree", "src", "dst", "input", "random"] )
    parser.add_argument('--edge_weight', default="freq", type=str, help="Edge weight configuraion", choices=["uniform", "freq", "random"])
    parser.add_argument('--bal', default="xbal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])
    parser.add_argument('--log_file',default='exp.csv',type=str,help='output log file')
    return parser.parse_args()

def get_partition_type(node_weight, edge_weight, bal):
    return f"n{node_weight}_e{edge_weight}_{bal}"
