"""Microbenchmarks for the roofline hardware rates (thesis Table tab:cm-microbench).

Measures, per hidden size h (fp32, b=4 bytes, s_H = 4h bytes per row):
  P_hat     one layer's GEMM on a sampled block: [n_rows, h] @ [h, h], TF32 off
  B_mem     neighbour gather with fanout e: mean over e scattered rows per dst
            (CSR SpMM = what DGL's copy_u/mean uses; index_select+sum also logged)
  B_net     NCCL all-to-all of s_H-byte rows across G GPUs (bytes received per GPU / s)

Pure PyTorch on synthetic blocks: no groot build, no graph data needed.

Usage (on the GPU box, never on WSL):
  python roofline_rates.py compute --out rates_compute.csv           # 1 GPU: P_hat, B_mem
  torchrun --nproc_per_node=4 roofline_rates.py net --tag nvlink   --out rates_net.csv
  NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 roofline_rates.py net --tag pcie_emu --out rates_net.csv
"""
import argparse
import csv
import os
import time

import torch

HS = [256, 512, 1024, 2048, 4096]
B = 4  # fp32


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters  # seconds per call


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def bench_compute(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(dev)
    rows = []
    for h in HS:
        s_h = h * B
        # --- P_hat: dst rows of a sampled block times a weight matrix ---
        x = torch.randn(args.n_dst, h, device=dev)
        w = torch.randn(h, h, device=dev)
        t = timeit(lambda: torch.mm(x, w))
        flops = 2.0 * args.n_dst * h * h
        rows.append(dict(gpu=gpu, rate="P_hat", kernel="gemm", h=h, n_dst=args.n_dst, fanout=args.fanout,
                         seconds=t, work=flops, value=flops / t / 1e12, unit="TFLOP/s"))
        del x, w

        # --- B_mem: mean over `fanout` scattered source rows per dst row ---
        # Source table sized to the block's src frontier, capped by memory.
        n_src = min(args.n_src, int(args.table_bytes // s_h))
        src = torch.randn(n_src, h, device=dev)
        nnz = args.n_dst * args.fanout
        col = torch.randint(0, n_src, (nnz,), device=dev)
        crow = torch.arange(0, nnz + 1, args.fanout, device=dev)
        val = torch.full((nnz,), 1.0 / args.fanout, device=dev)
        A = torch.sparse_csr_tensor(crow, col, val, size=(args.n_dst, n_src))
        gathered = float(nnz) * s_h  # S = alpha * e * s_H per dst, alpha=1
        t = timeit(lambda: torch.sparse.mm(A, src))
        rows.append(dict(gpu=gpu, rate="B_mem", kernel="csr_spmm", h=h, n_dst=args.n_dst, fanout=args.fanout,
                         seconds=t, work=gathered, value=gathered / t / 1e9, unit="GB/s"))
        t = timeit(lambda: src.index_select(0, col).view(args.n_dst, args.fanout, h).sum(1))
        rows.append(dict(gpu=gpu, rate="B_mem", kernel="index_select_sum", h=h, n_dst=args.n_dst,
                         fanout=args.fanout, seconds=t, work=gathered, value=gathered / t / 1e9, unit="GB/s"))
        del src, col, crow, val, A
        torch.cuda.empty_cache()
        print(rows[-3:], flush=True)
    append_rows(args.out, rows)


def bench_net(args):
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    rows = []
    for h in HS:
        s_h = h * B
        # Each GPU sends `rows_per_peer` rows to every other GPU (self-slice stays local).
        per_peer = args.rows_per_peer
        send = torch.randn(per_peer * world, h, device=dev)
        recv = torch.empty_like(send)
        t = timeit(lambda: dist.all_to_all_single(recv, send))
        received = float(per_peer * (world - 1)) * s_h  # bytes that actually crossed the link
        tt = torch.tensor([t], device=dev)
        dist.all_reduce(tt, op=dist.ReduceOp.MAX)
        t = tt.item()
        if rank == 0:
            rows.append(dict(gpu=torch.cuda.get_device_name(dev), rate="B_net", kernel=f"nccl_a2a_{args.tag}",
                             h=h, n_dst=per_peer * world, fanout=world, seconds=t, work=received,
                             value=received / t / 1e9, unit="GB/s"))
            print(rows[-1], flush=True)
        del send, recv
        torch.cuda.empty_cache()
    if rank == 0:
        append_rows(args.out, rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["compute", "net"])
    p.add_argument("--out", required=True)
    p.add_argument("--fanout", type=int, default=15)        # matches gen_roofline.F_OUT
    p.add_argument("--n_dst", type=int, default=100_000)    # dst rows of one block layer
    p.add_argument("--n_src", type=int, default=2_449_029)  # products |V|: scattered-read table
    p.add_argument("--table_bytes", type=float, default=20e9)  # cap so h=4096 fits on one A100
    p.add_argument("--rows_per_peer", type=int, default=25_000)
    p.add_argument("--tag", default="nvlink")
    a = p.parse_args()
    bench_compute(a) if a.mode == "compute" else bench_net(a)
