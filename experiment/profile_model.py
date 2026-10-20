"""
profile_model.py  (single GPU)  —  cross-check the simulator's compute cost.

Times, on one representative sampled batch:
  (A) FUSED  full-model forward + backward (what the real trainer does), and
  (B) SIM-STYLE per-layer isolated forward+backward summed (what
      simulation/common.py:measure_compute_cost_layer does).
Run normally to get CUDA-event times; run under `nsys profile --stats=true` to
confirm the fused GPU-kernel time matches the CUDA-event number (ground truth).

Usage:
  python3 profile_model.py --graph_name=products --data_dir=/root/graph \
      --fanouts=15,15,15 --hid_size=256 --model=sage --batch_size=1024
"""
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import dgl
from utils import get_args, Config
from utils.dataloading import get_feat_dim, load_topo
from nodepred import Sage, Gat

NUM_CLASSES = {"products": 47, "arxiv": 40, "papers100M": 172}


def cuda_time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/iter


def main():
    args = get_args()
    fanouts = [int(f) for f in args.fanouts.split(",")]
    device = 0
    torch.cuda.set_device(device)
    cfg = Config(graph_name=args.graph_name, world_size=1, num_partition=1, num_epoch=1,
                 fanouts=fanouts, batch_size=args.batch_size, system="profile", model=args.model,
                 cache_size="0GB", hid_size=args.hid_size, log_path="/tmp/p.csv",
                 data_dir=args.data_dir, nvlink=True, sample_mode="gpu")
    cfg.in_feat = get_feat_dim(cfg)
    nc = NUM_CLASSES.get(cfg.graph_name, 172)
    cfg.num_classes = nc

    graph, train_idx, _, _ = load_topo(cfg, is_pinned=False)
    graph = graph.to(device)
    train_idx = train_idx.to(device)

    # one representative batch of seeds
    seeds = train_idx[torch.randperm(train_idx.shape[0], device=device)[:args.batch_size]]

    Model = Sage if args.model == "sage" else Gat
    model = Model(cfg.in_feat, cfg.hid_size, len(fanouts), nc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- build the sampled blocks for the fused full-model step (seeds-out) ----
    blocks, cur = [], seeds
    for fo in reversed(fanouts):
        sg = dgl.sampling.sample_neighbors(graph, cur, fo)
        blk = dgl.to_block(sg, cur)
        blocks.insert(0, blk)
        cur = blk.srcdata[dgl.NID]
    in_feat = torch.rand((blocks[0].num_src_nodes(), cfg.in_feat), device=device)
    label = torch.randint(0, nc, (seeds.shape[0],), device=device)

    def fused_step():
        out = model(blocks, in_feat)
        loss = torch.nn.functional.cross_entropy(out, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    fused_ms = cuda_time(fused_step)

    # ---- sim-style: per-layer isolated fwd+bwd on each hop, summed ----
    from simulation.common import measure_compute_cost_layer
    layers = list(model.layers)

    def simstyle_step():
        total = 0.0
        cur2 = seeds
        for i, fo in enumerate(fanouts):
            sg = dgl.sampling.sample_neighbors(graph, cur2, fo)
            total += measure_compute_cost_layer(sg, layers[-i], cur2)  # mirrors the sim's indexing
            cur2 = torch.cat([sg.edges()[0], sg.edges()[1]]).unique()
        return total

    for _ in range(3):
        simstyle_step()
    torch.cuda.synchronize()
    sim_s = sum(simstyle_step() for _ in range(10)) / 10 * 1000  # ms/iter

    print(f"RESULT graph={cfg.graph_name} model={args.model} depth={len(fanouts)} "
          f"hid={cfg.hid_size} in_nodes={blocks[0].num_src_nodes()} | "
          f"FUSED_step_ms={fused_ms:.3f} | SIMSTYLE_step_ms={sim_s:.3f} | "
          f"ratio_sim/fused={sim_s/max(fused_ms,1e-9):.2f}", flush=True)

    # ---- feature-loading micro: real pinned-host -> GPU UVA gather vs PCIe model,
    #      swept over feature size to check linearity + achievable bandwidth. ----
    from dgl.utils import gather_pinned_tensor_rows
    frontier_ids = blocks[0].srcdata[dgl.NID].to(device)
    nf = frontier_ids.shape[0]
    feat_sizes = [int(x) for x in str(args.feat_sizes).split(",")] if getattr(args, "feat_sizes", None) else [64, 128, 256, 512, 1024]
    for fd in feat_sizes:
        host_feat = torch.empty((graph.num_nodes(), fd)).pin_memory()
        load_ms = cuda_time(lambda: gather_pinned_tensor_rows(host_feat, frontier_ids))
        bytes_moved = nf * fd * 4
        eff_bw = bytes_moved / (load_ms / 1000) / 1e9              # measured GB/s
        model_load_ms = bytes_moved / 16e9 * 1000                   # sim's PCIE_BW=16 GB/s
        print(f"FEATLOAD graph={cfg.graph_name} frontier={nf} feat={fd} "
              f"gather_ms={load_ms:.3f} eff_GBps={eff_bw:.1f} model16GBps_ms={model_load_ms:.3f} "
              f"per_node_us={load_ms*1000/nf:.4f}", flush=True)
        del host_feat


if __name__ == "__main__":
    main()
