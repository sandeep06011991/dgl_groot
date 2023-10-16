import pandas as pd
import re
OUT_DIR = "/home/sandeep/dgl_groot/python/"
FILENAME = f"{OUT_DIR}/experiment.csv"
import subprocess
import pandas as pd
import os

def check_single(ls):
    assert(len(ls)== 1)
    return ls[0]

def average(ls):
    return sum(ls)/len(ls)


def create_dataframe():
    d = {
        'graph': [],
        'model': [],
        'hidden_size': [],
        'batch_size': [],
        'fanout': [],
        'num_redundant_layers':[],
        'random_partition':[],
        'cache_per':[],
        'accuracy':[],
        'epoch_time':[],
        'sampling_time':[],
        'training_time':[],
        'max_cache_fraction':[]
     }
    df = pd.DataFrame(data=d)
    return df

def run_groot(graphname, model, cache_per, hidden_size,  minibatch_size, \
         fanout, num_redundant_layers, is_random_partition ):
    print(graphname, model, cache_per, hidden_size, minibatch_size, \
          fanout, num_redundant_layers, is_random_partition)
    cmd = ["python3",\
            "{}/run.py".format(OUT_DIR),\
        "--graph",graphname,  \
        "--model", model , \
        "--world_size", 4,\
        "--cache-rate" , str(cache_per),\
        "--hid_feat",  str(hidden_size), \
        "--batch", str(minibatch_size) ,\
        "--num-epochs", "5",\
        "--fan-out", fanout,\
        "--num_redundant_layers", num_redundant_layers,\
        "--random-partition", is_random_partition,\
        "--fanout", fanout
        ]

    output = subprocess.run(cmd, capture_output= True)

    # print(out,error)
    out = str(output.stdout)
    error = str(output.stderr)
    # print(out,error)
    if "out of memory" in error:
        epoch_time = "OOM"
        sampling_time = "OOM"
        training_time = "OOM"
        accuracy = "OOM"
        max_cache_fraction = "OOM"
        max_memory_used = "OOM"

    # #print("Start Capture !!!!!!!", graphname, minibatch_size)
    else:
        epoch_time = check_single(re.findall('epoch_time:(\d+\.\d+)',out))
        sampling_time = check_single(re.findall('sampling_time:(\d+\.\d+)', out))
        training_time = check_single(re.findall('training_time:(\d+\.\d+)',out))
        accuracy = check_single(re.findall('accuracy:(\d+\.\d+)',out))
        max_cache_fraction = check_single(re.findall('max_cache_fraction:(\d+\.\d+)',out))
        max_memory_used = average(re.findall('Max memory used:(\d+\.\d+)GB'))

    return {'graph':graphname, 'model':model, 'hidden_size':hidden_size, 'batch_size':minibatch_size, 'fanout':fanout,\
            'num_redundant_layers':num_redundant_layers, 'random_partition':is_random_partition, 'cache_per':max_cache_fraction,\
        'epoch_time':epoch_time, 'sampling_time':sampling_time, 'training_time':training_time, \
                'accuracy':accuracy, 'max_cache_fraction':max_cache_fraction, 'max_memory_used':max_memory_used}



def run_experiment_groot():
    if not os.path.isfile(FILENAME):
        df = create_dataframe()
    else:
        df = pd.read_csv(FILENAME)
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    models = ["gcn", "gat", "sage", "hgt"]
    hidden_sizes = [64, 128, 256, 1024]
    minibatch_sizes = [256, 512, 1024]
    graphs = ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]
    fanouts =  ["10,10,10", "20,20,20", "30,30,30"]

    graphs = ["ogbn-arxiv"]
    hidden_sizes = [256]
    models  = ['gat']
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



    for graph, model, hidden_size, minibatch_size, \
            fanout, num_redundant, is_random_partition in settings:
            cache_per = 0
            out = run_groot(graph, model, cache_per, hidden_size,  minibatch_size, \
                            fanout, num_redundant, is_random_partition )
            df1 = pd.DataFrame([out])
            cache_per = df['max_cache_per'].item()
            out = run_groot(graph, model, cache_per, hidden_size,  minibatch_size, \
                            fanout, num_redundant, is_random_partition )
            df2 = pd.DataFrame([out])
            df = pd.concat([df,df1,df2])
            df.to_csv(FILENAME)




if __name__ == "__main__":
    run_experiment_groot()
