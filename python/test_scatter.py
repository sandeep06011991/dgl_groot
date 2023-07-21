import torch
import dgl
from dgl.groot.entry import test


data = torch.tensor([1,2,3,4,5,6,7,8], dtype = torch.int32).to(0)
p_map = torch.tensor([0,2,1,0,1,0,1,0], dtype = torch.int32).to(0)
num_parts = 3
print(torch.device(data.device), torch.device(0))
obj = test(data, p_map, num_parts)
print(type(obj.index))
print(obj.sizes)
################## Three main elements ##################
# part_1 = [1,4,6,8]
# inverse_1 = [0,3,5,7]

# part_2 = [3,5,7]
# inverse_2 = []

# part_3 = [2]
# inverse_3 = [1]
################# Three main elements ####################
################# Algorithm ##############################
