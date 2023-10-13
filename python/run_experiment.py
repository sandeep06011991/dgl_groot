OUT_DIR = "/home/sandeep/dgl_groot/python/"
import subprocess
import pandas
def get_gpu_memory_used():
    from pynvml import *
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

def get_max_cache():
    return max_cache_dict
    # pandas.read('table')
    # exists max_avalable memory
    #   Cache the graph for that memory size

def create_pandas():
    import pandas as pd
    d = {
        'graph': [],
        'model': [],
        'hidden_size': [],
        'batch_size': [],
        'fanout': [],
        'num_redundant_layers':[],
        'random_partition':[]
     }
    df = pd.DataFrame(data=d)

def run_groot(graphname, model, cache_per, hidden_size,  minibatch_size, \
         fanout, num_redundant_layers, is_random_partition ):
    print(graphname, model, cache_per, hidden_size, minibatch_size, \
          fanout, num_redundant_layers, is_random_partition)
    cmd = ["python3",\
            "{}/run.py".format(OUT_DIR),\
        "--graph",graphname,  \
        "--model", model , \
        "--cache-per" , str(cache_per),\
        "--hid_feat",  str(hidden_size), \
        "--batch", str(minibatch_size) ,\
        "--num-epochs", "5",\
        "--fan-out", fanout,\
        "--num_redundant_layers", num_redundant_layers,
        "--random-partition", is_random_partition
        ]

    output = subprocess.run(cmd, capture_output= True)

    # print(out,error)
    out = str(output.stdout)
    error = str(output.stderr)
    # print(out,error)
    if "out of memory" in error:
        return {"sample_get":"OOM", "feat_movement":"OOM", \
                "training_time": "OOM", "epoch_time": "OOM", "epoch": "OOM", \
                "accuracy": "OOM", "data_moved_feat": "OOM", "data_moved_hidden":"OOM", "edges_moved": "OOM"}

    # #print("Start Capture !!!!!!!", graphname, minibatch_size)
    # try:
    # # if True:
    #     accuracy  = check_single(re.findall("accuracy:(\d+\.\d+)",out))
    #     epoch = check_single(re.findall("epoch_time:(\d+\.\d+)",out))
    #     sample_get  = check_single(re.findall("sample_time:(\d+\.\d+)",out))
    #     movement_graph =  check_single(re.findall("movement graph:(\d+\.\d+)",out))
    #     movement_feat = check_single(re.findall("movement feature:(\d+\.\d+)",out))
    #     forward_time = check_single(re.findall("forward time:(\d+\.\d+)",out))
    #     backward_time = check_single(re.findall("backward time:(\d+\.\d+)",out))
    #     data_moved = check_single(re.findall("data movement:(\d+\.\d+)MB",out))
    #     edges_moved = re.findall("edges per epoch:(\d+\.\d+)",out)
    #     s = []
    #     if(num_partition == -1):
    #         num_partition = 4
    #     for i in range(num_partition):
    #         s.append(float(edges_moved[i]))
    #     edges_moved_avg = sum(s) / num_partition
    #     edge_moved_max = max(s)
    #     edge_moved_skew = (max(s) - min(s)) /min(s)
    #     sample_get = "{:.2f}".format(float(sample_get))
    #     movement_graph = "{:.2f}".format(float(movement_graph))
    #     movement_feat = "{:.2f}".format(float(movement_feat))
    #     accuracy = "{:.2f}".format(float(accuracy))
    #     forward_time = "{:.2f}".format(float(forward_time))
    #     epoch = "{:.2f}".format(float(epoch))
    #     backward_time = "{:.2f}".format(float(backward_time))
    #     data_moved = int(float(data_moved))
    #     # edges_moved = int(float(edges_moved))
    #
    # except Exception as e:
    #     with open('exception_occ.txt','w') as fp:
    #         fp.write(error)
    #
    #     sample_get = "error"
    #     movement_graph = "error"
    #     movement_feat = "error"
    #     forward_time = "error"
    #     backward_time = "error"
    #     accuracy = "error"
    #     epoch = "error"
    #     data_moved = "error"
    #     edges_moved = "error"
    # return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
    #         "movement_graph":movement_graph, "movement_feat": movement_feat, "epoch":epoch,
    #             "accuracy": accuracy, "data_moved":data_moved, "edge_moved_avg":edges_moved_avg,\
    #                 "edge_moved_max": edge_moved_max, "edge_moved_skew":edge_moved_skew}



def run_experiment_groot():
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    models = ["gcn", "gat", "sage", "hgt"]
    hidden_sizes = [64, 128, 256, 1024]
    minibatch_sizes = [256, 512, 1024]
    graphs = ["ogbn-arxiv"]
    fanouts =  ["15,10", "15,10,5", "10,10,10", "20,20,20", "30,30,30"]
    # Flatten everything
    settings = []
    for graph in graphs:
        for model in models:
            for hidden_size in hidden_sizes:
                for minibatch_size in minibatch_sizes:
                    for fanout in fanouts:
                        for num_redundant in range(0,len(fanout.split(",")) + 1):
                            for is_random_partition in [True,False]:
                                settings.append((graph, model, hidden_size, minibatch_size,\
                                                fanout, num_redundant, is_random_partition))



    with open(OUT_DIR + '/groot_{}.txt'.format(SYSTEM),'a') as fp:
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size |"+\
                "num_partitions | num-layers |" + \
            " model  | fanout |  sample_get | move-graph | move-feature | forward | backward  |"+\
                " epoch_time | accuracy | data_moved | edges_computed\n")

    for graph, model, hidden_size, minibatch_size,fanout, num_redundant, is_random_partition in settings:
            out = run_occ(graphname, model,  cache, hidden_size, fsize,\
                    batch_size, num_layers, num_partition, fanout)
            with open(OUT_DIR + '/groot_{}.txt'.format(SYSTEM),'a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |{} | {} \n".\
                format(graphname , SYSTEM, cache, hidden_size, fsize, batch_size,\
                    num_partition, num_layers, model, fanout, out["sample_get"], \
                    out["movement_graph"], out["movement_feat"], out["forward"], out["backward"], \
                     out["epoch"], out["accuracy"], out["data_moved"], out["edges_moved"]))




if __name__ == "__main__":
    run_experiment_groot()
    # run_experiment_occ("gat-pull")
