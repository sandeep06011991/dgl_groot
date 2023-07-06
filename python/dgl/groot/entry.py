from .. import backend as F
from .._ffi.function import _init_api


def wrap_function():
    a = _CAPI_testffi()
    print(a)
    
_init_api("dgl.groot", __name__)
