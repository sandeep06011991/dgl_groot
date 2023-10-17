import argparse
from exp.util import *
from exp.groot_trainer import  bench_groot_batch

def get_configs(graph_name, system, log_path, data_dir):
    fanouts = [[20,20,20],[30,30,30],[40,40,40]]
    batch_sizes = [512,1024, 4096]
    models = ["gat","sage"]
    hid_sizes = [256,512, 1024]

    fanouts = [[10,10,10]]
    batch_sizes = [4096]
    models = ['gat']
    hid_sizes = [256]
    configs = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            for model in models:
                for hid_size in hid_sizes:
                    for num_redundant_layers in range(len(fanout) + 1):
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
    configs = get_configs(graph_name="ogbn-products", system="groot-uva", log_path="./log/groot.csv", data_dir="/data/ogbn/processed/")
    bench_groot_batch(configs=configs, test_acc=True)
    # configs = get_configs(graph_name="ogbn-papers100M", system="uva", log_path="./log/groot.csv", data_dir="/data/ogbn/processed/")
    # bench_groot_batch(configs=configs, test_acc=True)
