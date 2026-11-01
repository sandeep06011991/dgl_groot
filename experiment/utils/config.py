import os
import pandas as pd

class Config:
    def __init__(self, graph_name, world_size, num_partition, num_epoch, fanouts,
                 batch_size, system, model, hid_size, cache_size, log_path, data_dir,
                 nvlink = False, partition_type="ndst_efreq_xbal", sample_mode="uva"):
        try:
            self.machine_name = os.environ['MACHINE_NAME']
        except Exception as e:
            self.machine_name = "jupiter"
        self.graph_name = graph_name
        self.world_size = world_size
        self.num_partition = num_partition
        self.num_epoch = num_epoch
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.system = system
        self.model = model
        self.cache_size = cache_size
        self.hid_size = hid_size
        self.log_path = log_path
        self.data_dir = data_dir
        self.nvlink = nvlink
        self.partition_type = partition_type
        self.sample_mode = sample_mode
        self.num_redundant_layer = 0
        self.cache_frac = -1.0   # quiver cache as a fraction of feat (<=0 = n/a; cache_size used instead)
        self.in_feat = -1
        self.num_classes = -1
        self.test_model_acc = False
        
    def get_file_name(self):
        if "groot" not in self.system:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                     f"{len(self.fanouts)}x{self.fanouts[0]}_{self.cache_size}")
        else:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                    f"{len(self.fanouts)}x{self.fanouts[0]}_{self.num_redundant_layer}_{self.cache_size}")

    def header(self):
        return ["timestamp","machine_name", "hardware", "graph_name", "feat_width", "world_size", "num_partition", "num_epoch", "fanouts", "num_redundant_layers", \
                "batch_size", "system", \
                    "model", "hid_size", "cache_size", "cache_frac", "partition_type"]

    def content(self):
        connection = "_nvlink" if self.nvlink else "_pcie"
        machine_name = self.machine_name + connection
        hardware = "nvlink" if self.nvlink else "pcie_emu"
        return [pd.Timestamp('now'), machine_name, hardware, self.graph_name, self.in_feat, self.world_size, self.num_partition, self.num_epoch, self.fanouts, self.num_redundant_layer, \
                    self.batch_size, self.system, self.model, self.hid_size, self.cache_size, self.cache_frac, self.partition_type]

    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += f"num_classes={self.num_classes}"
        res += "\n"
        return res    


# def get_default_config(graph_name, system, log_path, data_dir, num_redundant_layer = 0):
#     configs = []
#     partitioning_graph = ""
#     balancing="edge"
#     training_nodes="xbal"
#     for model in ["sage", "gat"]:
#        config = Config(graph_name=graph_name,
#                        world_size=4,
#                        num_epoch=5,
#                        fanouts=[20,20,20],
#                        batch_size=1024,
#                        system=system,
#                        model=model,
#                        cache_size = 0,
#                        hid_size=256,
#                        log_path=log_path,
#                        data_dir=data_dir)
       
#        config.num_redundant_layer = num_redundant_layer
#        config.partition_type = f"{partitioning_graph}_w4_{balancing}_{training_nodes}"
#        configs.append(config)
#     return configs
