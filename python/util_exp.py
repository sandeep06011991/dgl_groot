import pandas as pd
import pandas 

def clean_and_sort(filename):
    df = pd.read_csv(filename)

    is_duplicate = df.duplicated(subset=['machine_name', 'graph_name', 'world_size', 'num_epoch', 'fanouts',
                                         'num_redundant_layers', 'batch_size', 'system', 'model', 'hid_size',
                                         'cache_rate', 'partition_type'], keep = 'last')

    unique_df = df[~ is_duplicate]

    sorted_df = unique_df.sort_values(['machine_name', 'graph_name', 'world_size', 'num_epoch', 'fanouts',
                                   'batch_size','model', 'hid_size',
                                   'system', 'num_redundant_layers', 'cache_rate', 'partition_type'])
    with open(filename,'w') as fp:
        fp.write(sorted_df.to_csv(index = False))
    print("removed", is_duplicate.sum(), "from", filename)

def merge():
    for filename in ["batch_size.csv","depth.csv","hidden_size.csv",\
                 "partitions.csv","redundant_layers.csv"]:
# filename = "default.csv"
        df1 = pandas.read_csv('log/' + filename)
        df2 = pandas.read_csv('temp/' + filename )
        #df3 = pandas.read_csv('log-run1/' + filename)
        df = pandas.concat([df1,df2])
        with open('log/' + filename,'w') as fp:
            fp.write(df.to_csv(index = False))
        print("Done", filename)

from io import StringIO
def correction(filename):
    with open(filename,'r') as fp:
        for line in fp.readlines():
            csvStringIO = StringIO(line)
            df = pd.read_csv(csvStringIO, sep=",", header=None)
            print(len(df.cols))

            
if __name__ == "__main__":
    # correction('temp/default.csv')
    # merge()
    filenames= ["batch_size", "depth", "partitions", \
       "default", "hidden_size", "redundant_layers"]
    #filenames = ["default"]
    for filename in filenames:
       clean_and_sort(f'log/{filename}.csv')
