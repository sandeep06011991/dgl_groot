import pandas as pd

def clean_and_sort(filename):
    df = pd.read_csv(filename)

    is_duplicate = df.duplicated(subset=['machine_name', 'graph_name', 'world_size', 'num_epoch', 'fanouts',
                                         'num_redundant_layers', 'batch_size', 'system', 'model', 'hid_size',
                                         'cache_rate', 'partition_type'], keep = 'last')

    unique_df = df[~ is_duplicate]

    sorted_df = unique_df.sort_values(['machine_name', 'graph_name', 'world_size', 'num_epoch', 'fanouts',
                                   'batch_size','model', 'hid_size',
                                   'system', 'num_redundant_layers', 'cache_rate', 'partition_type'])
    with open(filename+'_no_dup.csv','w') as fp:
        fp.write(sorted_df.to_csv(index = False))
    print("removed", is_duplicate.sum(), "from", filename)

if __name__ == "__main__":
    filenames= ["batch_size", "depth", "partitions", \
        "default", "hidden_size", "redundant_layers"]
    #filenames = ["default"]
    for filename in filenames:
        try:
            clean_and_sort(f'log/{filename}.csv')
        except:
            pass
