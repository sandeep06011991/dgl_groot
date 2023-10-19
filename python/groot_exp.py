import argparse
from exp.util import *
from exp.groot_trainer import  bench_groot_batch

def get_configs(graph_name, system, log_path, data_dir):
    fanouts = [[20,20,20],[20,20,20,20]]
    batch_sizes = [1024, 4096]
    models = ["gat","sage"]
    hid_sizes = [256,512]
    configs = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            for model in models:
                for hid_size in hid_sizes:
                    for num_redundant_layers in range(1, len(fanout) + 1):
                        config = Config(graph_name=graph_name,
                                        world_size=4,
                                        num_epoch=10,
                                        fanouts=fanout,
                                        batch_size=batch_size,
                                        system=system,
                                        model=model,
                                        cache_rate = 0,
                                        hid_size=hid_size,
                                        log_path=log_path,
                                        data_dir=data_dir,)
                        config.num_redundant_layer = num_redundant_layers
                        configs.append(config)
    return configs

if __name__ == "__main__":
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    # configs = get_configs(graph_name="ogbn-products", system="groot-uva", log_path="./log/groot.csv", data_dir="/data/ogbn/processed/")
    # bench_groot_batch(configs=configs, test_acc=True)
    # print("prod uva done !")
    configs = get_configs(graph_name="ogbn-papers100M", system="groot-uva", log_path="./log/groot.csv", data_dir="/data/ogbn/processed/")
    bench_groot_batch(configs=configs, test_acc=True)
    print("paper uva done !")
    configs = get_configs(graph_name="ogbn-products", system="groot-gpu", log_path="./log/groot.csv", data_dir="/data/ogbn/processed/")
    bench_groot_batch(configs=configs, test_acc=True)
    print("prod gpu done ")