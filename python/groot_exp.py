import argparse
from exp.util import *
from exp.groot_trainer import  bench_groot_batch
from exp.dgl_trainer import bench_dgl_batch
from exp.quiver_trainer import bench_quiver_batch
import quiver
class DEFAULT_SETTING:
    batch_size = 1024
    hid_size =128
    fanouts = [30,30,30]
    models = [ "gat","sage"]
    num_redundant_layers = 0

DATA_DIR_OGBN = "/data/ogbn/processed/"
DATA_DIR_SNAP = "/data/snap/"

def get_default_config(graph_name, system, log_path, data_dir):
    configs = []
    for model in DEFAULT_SETTING.models:
       config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=5,
                       fanouts=DEFAULT_SETTING.fanouts,
                       batch_size=DEFAULT_SETTING.batch_size,
                       system=system,
                       model=model,
                       hid_size=DEFAULT_SETTING.hid_size,
                       log_path=log_path,
                       data_dir=data_dir,)
       config.num_redundant_layer = 0
       configs.append(config)
    return configs

def get_number_of_redundant_layers(graph_name, system, log_path, data_dir):
    fanout = DEFAULT_SETTING.fanouts
    configs = []
    for model in DEFAULT_SETTING.models:
        for num_redundant_layers in range(0, len(fanout) + 1):
            config = Config(graph_name=graph_name,
                            world_size=4,
                            num_epoch=5,
                            fanouts=fanout,
                            batch_size=DEFAULT_SETTING.batch_size,
                            system=system,
                            model=model,
                            cache_rate = 0,
                            hid_size=DEFAULT_SETTING.hid_size,
                            log_path=log_path,
                            data_dir=data_dir,)
            config.num_redundant_layer = num_redundant_layers
            configs.append(config)
    return configs

def get_partition_type(graph_name, system, log_path, data_dir):
    configs = []
    for model in DEFAULT_SETTING.models:
        for partition_type in ["edge_balanced", "node_balanced", "random"]:
            config = Config(graph_name=graph_name,
                            world_size=4,
                            num_epoch=5,
                            fanouts=DEFAULT_SETTING.fanouts,
                            batch_size= DEFAULT_SETTING.batch_size,
                            system=system,
                            model=model,
                            hid_size=DEFAULT_SETTING.hid_size,
                            log_path=log_path,
                            data_dir=data_dir,)
            config.partition_type = partition_type
            configs.append(config)
    return configs

def get_depth_config(graph_name, system, log_path, data_dir):
    fanouts = [[10,10,10],[10,10,10,10],[10,10,10,10,10]]
    configs = []
    for fanout in fanouts:
        for model in DEFAULT_SETTING.models:
            config = Config(graph_name=graph_name,
                            world_size=4,
                            num_epoch=10,
                            fanouts=fanout,
                            batch_size=DEFAULT_SETTING.batch_size,
                            system=system,
                            model=model,
                            hid_size=DEFAULT_SETTING.hid_size,
                            log_path=log_path,
                            data_dir=data_dir,)
            config.num_redundant_layer = DEFAULT_SETTING.num_redundant_layers
            configs.append(config)
    return configs

def get_batchsize_config(graph_name, system, log_path, data_dir):
    batch_sizes = [1024, 4096, 4096 * 4]
    configs = []
    for batch_size in batch_sizes:
        for model in DEFAULT_SETTING.models:
            config = Config(graph_name=graph_name,
                            world_size=4,
                            num_epoch=10,
                            fanouts=DEFAULT_SETTING.fanouts,
                            batch_size=batch_size,
                            system=system,
                            model=model,
                            hid_size=DEFAULT_SETTING.hid_size,
                            log_path=log_path,
                            data_dir=data_dir,)
            config.num_redundant_layer = DEFAULT_SETTING.num_redundant_layers
            configs.append(config)
    return configs 

def get_hidden_config(graph_name, system, log_path, data_dir):
    hidden_sizes = [64, 128, 256, 512,]
    configs = []
    for hidden_size in hidden_sizes:
        for model in DEFAULT_SETTING.models:
            config = Config(graph_name=graph_name,
                            world_size=4,
                            num_epoch=10,
                            fanouts=DEFAULT_SETTING.fanouts,
                            batch_size=DEFAULT_SETTING.batch_size,
                            system=system,
                            model=model,
                            hid_size=hidden_size,
                            log_path=log_path,
                            data_dir=data_dir,)
            config.num_redundant_layer = DEFAULT_SETTING.num_redundant_layers
            configs.append(config)
    return configs 

def get_abalation_config(graph_name, system, log_path, data_dir):
    configs = []
    configs.extend(get_depth_config(graph_name, system, log_path, data_dir))
    configs.extend(get_batchsize_config(graph_name, system, log_path, data_dir))
    configs.extend(get_hidden_config(graph_name, system, log_path, data_dir))
    return configs

def get_configs(graph_name, system, log_path, data_dir):
    fanouts = [[20,20,20],[20,20,20,20],[20,20,20,20,20]]
    batch_sizes = [1024, 4096]
    models = ["gat","sage"]
    hid_sizes = [256,512]
    configs = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            for model in models:
                for hid_size in hid_sizes:
                    for num_redundant_layers in range(0, len(fanout) + 1):
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

def best_configuration():
    for graph_name in ["ogbn-products", "ogbn-papers100M"]:
        configs = get_number_of_redundant_layers(graph_name = graph_name,\
               system = "groot", log_path='./log/redundant_layers.csv', data_dir="/data/ogbn/processed/")
        bench_groot_batch(configs = configs, test_acc = True)
        # bench_dgl_batch(configs = [configs[0]], test_acc = True)
        configs = get_partition_type(graph_name = graph_name, system = "groot",\
                                    log_path='./log/partitions.csv', \
                                     data_dir="/data/ogbn/processed/")
        bench_groot_batch(configs = configs, test_acc = True)

def quiver_experiment(graph_name: str):
        configs = []
        # config = get_default_config(graph_name, system="quiver", log_path = "./log/default.csv", \
        #                          data_dir="/data/ogbn/processed/")
        # configs.append(config)
        config = get_batchsize_config(graph_name= graph_name, system="quiver", log_path="./log/batch_size.csv", data_dir="/data/ogbn/processed/")
        configs.append(config)
        configs = get_hidden_config(graph_name=graph_name, system="quiver", log_path="./log/hidden_size.csv",
                                    data_dir="/data/ogbn/processed/")
        configs.append(config)
        configs = get_depth_config(graph_name=graph_name, system="quiver", log_path="./log/depth.csv",
                                   data_dir="/data/ogbn/processed/")
        configs.append(config)
        bench_quiver_batch(configs = configs, test_acc = True )

def experiment_one():
    for graph_name in [ "ogbn-products"]:
        if "ogbn" in graph_name:
            configs = get_default_config(graph_name, system="default", log_path = "./log/default.csv", \
                                         data_dir=DATA_DIR_OGBN)
            test_acc = True
        else:
            configs = get_default_config(graph_name, system="default", log_path = "./log/default.csv", \
                                         data_dir=DATA_DIR_SNAP)
            test_acc = False
        bench_dgl_batch(configs=configs, test_acc=test_acc)
        bench_quiver_batch(configs = configs, test_acc = test_acc )
        bench_groot_batch(configs=configs, test_acc=test_acc )
    return
    # #return
    # print("Default experiment done")
    # for graph_name in ["ogbn-papers100M"]:
    #     configs = get_batchsize_config(graph_name= graph_name, system="batch_size", log_path="./log/batch_size.csv", data_dir="/data/ogbn/processed/")
    #     bench_dgl_batch(configs=configs, test_acc =True )
    #     bench_groot_batch(configs=configs, test_acc=True )
    #     #bench_quiver_batch(configs = configs, test_acc = False )
    # print("Batch experiment done")
    # for graph_name in ["ogbn-products", "ogbn-papers100M"]:
    #     configs = get_hidden_config(graph_name=graph_name, system="hidden_size", log_path="./log/hidden_size.csv",
    #                                 data_dir="/data/ogbn/processed/")
    #     bench_dgl_batch(configs=configs, test_acc=True )
    #     bench_groot_batch(configs=configs, test_acc=True )
    #     #bench_quiver_batch(configs = configs, test_acc = False )
    # print("Hidden Experiment Done")
    for graph_name in ["ogbn-products"]:
        configs = get_depth_config(graph_name=graph_name, system="depth", log_path="./log/depth.csv",
                                   data_dir="/data/sandeep/groot_data/ogbn-processed/")
        bench_dgl_batch(configs=configs, test_acc=True )
        # bench_groot_batch(configs=configs, test_acc=True)
        #bench_quiver_batch(configs = configs, test_acc = False )
    print("Hidden Experiment Done")

def max_memory_measurement():
    configs = []
    for graph_name in ["ogbn-products"]:
        config = get_depth_config(graph_name=graph_name, system="dgl", log_path="./log/memory.csv",
                                   data_dir="/data/sandeep/groot_data/ogbn/processed/")
        configs.extend(config)
        # configs = get_hidden_config(graph_name=graph_name, system="dgl", log_path="./log/memory.csv",
        #                             data_dir="/data/sandeep/groot_data/ogbn/processed/")
        # configs.extend(config)
        # configs = get_batchsize_config(graph_name=graph_name, system="dgl", log_path="./log/memory.csv",
        #                             data_dir="/data/sandeep/groot_data/ogbn/processed/")
        # configs.extend(config)
        bench_quiver_batch(configs=configs, test_acc=True)
        # bench_groot_batch(configs=configs, test_acc=True)
        #bench_quiver_batch(configs = configs, test_acc = False )
        print("Memory measurements", graph_name)
if __name__ == "__main__":
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    quiver.init_p2p(device_list=list(range(4)))
    experiment_one()
    #best_configuration()
    # all_experiments()
    # all_experiments()

    # max_memory_measurement()