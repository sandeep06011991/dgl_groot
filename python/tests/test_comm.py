import dgl
import torch 


# Test 
# Tesnor A_gpu = torch.ones(4) * A_gpu
# Operator Shuffle called from all gpus in multiprocess
# Tensor B = Shuffle(A_gpu, [0,1,2,3])
# B = torch.tensor([0,1,2,3])
def test_tensor_shuffle():
    pass 



if __name__ == "__main__":
    test_tensor_shuffle()