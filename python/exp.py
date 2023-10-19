import argparse
from exp.util import *
from exp.dgl_trainer import bench_dgl_batch
from exp.quiver_trainer import bench_quiver_batch
import torch.multiprocessing as mp

def get_configs(graph_name, system, log_path, data_dir):
    fanouts = [[10, 10, 10]]
    batch_sizes = [4096]
    models = ["sage"]
    hid_sizes = [128]
    cache_rates = [0]
    # fanouts = [[20,20,20],[20,20,20,20], [30,30,30]]
    # batch_sizes = [1024, 4096]
    # models = ['gat', 'sage']
    # hid_sizes = [256, 512]
    configs = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            for model in models:
                for hid_size in hid_sizes:
                    for cache_rate in cache_rates:
                        config = Config(graph_name=graph_name, 
                                        world_size=4, 
                                        num_epoch=10, 
                                        fanouts=fanout, 
                                        batch_size=batch_size, 
                                        system=system, 
                                        model=model,
                                        hid_size=hid_size, 
                                        cache_rate=cache_rate, 
                                        log_path=log_path,
                                        data_dir=data_dir)
                        configs.append(config)
    return configs

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # configs = get_configs(graph_name="ogbn-papers100M", system="uva-dgl", log_path="./log/dgl.csv", data_dir="/data/ogbn/processed/")
    # bench_dgl_batch(configs=configs, test_acc=True)
    # configs = get_configs(graph_name="ogbn-papers100M", system="uva-dgl", log_path="./log/dgl.csv", data_dir="/data/ogbn/processed/")
    # bench_dgl_batch(configs=configs, test_acc=True)
    # configs = get_configs(graph_name="ogbn-products", system="gpu", log_path="./log/dgl.csv", data_dir="/data/ogbn/processed/")
    # bench_dgl_batch(configs=configs, test_acc=True)
    SYSTEM = "quiver"
    configs = get_configs(graph_name="ogbn-arxiv", system=f"gpu-{SYSTEM}", log_path=f"./log/{SYSTEM}.csv", data_dir="/data/ogbn/processed/")
    bench_quiver_batch(configs=configs, test_acc=True)
