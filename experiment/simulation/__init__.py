from .hybrid_dp_sim import simulate_optimal_k
from .hybrid_pipe_sim import get_costs_for_pipeline_parallelism, get_costs_for_split_parallelism

# `simulate` (the workload counter) pulls in dgl.dev (compiled groot C++). The
# cost-model simulators above only use upstream DGL ops, so import `simulate`
# lazily to keep `from simulation import simulate_optimal_k` working on a plain
# DGL install (no groot build required for the cost model).
def simulate(*args, **kwargs):
    from .simulate import simulate as _simulate
    return _simulate(*args, **kwargs)