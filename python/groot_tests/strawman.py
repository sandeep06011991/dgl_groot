import torch.cuda
import os
import torch.distributed as dist

import dgl
import  dgl.backend as F
import torch.multiprocessing as mp
from dgl.groot import *
from dgl.groot.models import *
from dgl._ffi.function import _init_api
import dgl
import  dgl.backend as F
import torch.multiprocessing as mp
from dgl.groot import *
from dgl.groot.models import *
from dgl._ffi.function import _init_api
import torch.cuda
import os
import torch.distributed as dist
import numpy as np

def test_me():
    test()
    print("test ok!")

if __name__ == "__main__":
    test_me()