import dgl
from dgl.groot import ScatterArray
import torch

def test_array_scatter():
    data = torch.tensor([10,9,8,7,6,5,4])
    p_map = torch.tensor([0,1,0,1,0,1,0])
    num_partitions = 2
    print(ScatterArray(data, p_map, num_partitions))


if __name__ == "__main__":
    test_array_scatter()
