#!/usr/bin/env python3
"""Sampling-only per-layer counts for the classic roofline (dissertation Sec. 3).

Replays the per-GPU sampling of the three trained systems -- without features,
model or training -- and writes, for every (system, k, minibatch, layer), ONE ROW of
aggregates over the G GPUs. These are the inputs of Eq. (cm-point);
defense_latex/gen_roofline_points.py turns them into roofline points.

Systems (column `system`, split point `k`; layers l counted FROM THE SEEDS):
  split     gsplit / data-parallel hybrid (SplitGraphLoader, split_trainer.py),
            k = 0 (gsplit) ... L (every layer data-parallel)
  pipeline  gsplit_pipepar (pipeline_split_trainer.py), k = 0 only (a data-parallel prefix
            has no shuffle to overlap). One step runs
            two micro-batches of batch_size/2; only ONE is sampled, and the two are
            assumed identical (balanced; same redundancy and communication).
  dgl       DGL baseline (GraphDataloader, dgl_trainer.py), pure data parallelism.

Seeds and batch sizes (as the loaders draw them):
  split, pipeline  batch_size is GLOBAL: steps = |train| // bs, and GPU q draws
                   |train of partition q| // steps of its own partition's train nodes
                   (bs = batch_size for split, batch_size/2 for a pipeline micro-batch)
  dgl              batch_size / G per GPU, from GPU q's contiguous 1/G slice of train_idx

Per layer, per GPU q (same loop as SplitSampler::sampleOneBatch, split_sampler.h):
  sample fanout[l] in-neighbours of q's frontier; src_q = unique(frontier_q + neighbours)
  l <  k  (dp)     q's next frontier = src_q; nothing is shuffled
  l >= k  (split)  src_q is scattered to the owners: q receives the rows of src_q owned
                   by other GPUs; q's next frontier = nodes q owns that any GPU requested
  (so at l = k of a hybrid the destination rows are still the redundant DP frontier)

Columns (max = largest GPU, each column maximised on its own; sum = over GPUs):
  n_max, n_sum   destination rows computed in layer l
  e_max, e_sum   edges aggregated in layer l
  m_max, m_sum   nodes shuffled by split parallelism in layer l: rows a GPU receives
                 from their owners (0 in dp layers)
  src_max        unique source rows of the largest block
  u              distinct destination rows of layer l over all GPUs
                 (redundant computation = n_sum - u)

Notes: fanouts[l] is applied at hop l from the seeds (reverse of the trainers'
input-first order; only matters for non-uniform fanouts). The outermost split layer
really shuffles input features (width F_in), not hidden rows (assumption M1).

Runs on CPU (no GPU kernels) but needs DGL + the prepared graph: run it next to the
data. Example:
    python3 experiment/roofline/sample_counts.py --data_dir=/root/graph \
        --graph_name=products --model=sage --fanouts=15,15,15,15 \
        --batch_size=1024 --world_size=4 --num_batches=20 \
        --out=defense_latex/results/roofline/products_sage_4x15_b1024.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import dgl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_dir", required=True, help="dir holding <graph_name>/ and partition_map/")
    p.add_argument("--graph_name", default="products")
    p.add_argument("--model", default="sage", choices=["sage", "gat"],
                   help="gat adds self loops, as load_topo does")
    p.add_argument("--fanouts", default="15,15,15,15", help="comma list, hop 0 = seeds")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="GLOBAL minibatch, as passed to train_main.py --batch_size")
    p.add_argument("--world_size", type=int, default=4, help="number of GPUs = partitions")
    p.add_argument("--partition_type", default="ndst_efreq_xbal")
    p.add_argument("--num_batches", type=int, default=20, help="sampled minibatches per strategy")
    p.add_argument("--seed", type=int, default=0, help="seeds torch and DGL RNGs")
    p.add_argument("--out", required=True, help="output CSV path")
    return p.parse_args()


def load(args):
    d = os.path.join(args.data_dir, args.graph_name)
    sym = "sym" if args.graph_name in ["orkut", "friendster"] else "xsym"
    indptr = torch.from_numpy(np.load(os.path.join(d, f"indptr_{sym}.npy")))
    indices = torch.from_numpy(np.load(os.path.join(d, f"indices_{sym}.npy")))
    g = dgl.graph(("csc", (indptr, indices, torch.empty(0, dtype=indices.dtype))))
    if args.model == "gat" and sym == "xsym":
        g = dgl.add_self_loop(g)
    train_idx = torch.from_numpy(np.load(os.path.join(d, "train_idx.npy"))).long()
    pm_path = os.path.join(args.data_dir, "partition_map", args.graph_name,
                           f"{args.graph_name}_w{args.world_size}_{args.partition_type}.npy")
    owner = torch.from_numpy(np.load(pm_path)).long()
    return g, train_idx, owner


class SeedDrawer:
    """Per-GPU seed sets for one minibatch, as each system's loader draws them."""

    def __init__(self, train_idx, owner, G, batch_size):
        self.G, self.B = G, batch_size
        self.part_train = [train_idx[owner[train_idx] == q] for q in range(G)]
        loc = train_idx.numel() // G + 1                       # dataloader.py set_target_idx
        self.dgl_slices = [train_idx[q * loc:(q + 1) * loc] for q in range(G)]
        self.n_train = train_idx.numel()

    @staticmethod
    def _draw(pool, k):
        return pool[torch.randperm(pool.numel())[:k]]

    def split_loader(self, bs):
        """SplitGraphLoader.set_target_idx: bs is global."""
        steps = self.n_train // bs
        return [self._draw(p, p.numel() // steps) for p in self.part_train]

    def dgl_loader(self):
        """GraphDataloader: batch_size // G per GPU from its contiguous slice."""
        return [self._draw(s, self.B // self.G) for s in self.dgl_slices]


def run_system(g, owner, seeds, fanouts, k, G):
    """Replay SplitSampler::sampleOneBatch for every GPU. Returns one stats dict per layer."""
    frontiers = list(seeds)
    rows = []
    for l, fanout in enumerate(fanouts):
        n, e, m, src = [], [], [], []
        next_parts = [[] for _ in range(G)]           # pieces of each GPU's next frontier
        for q in range(G):
            sg = dgl.sampling.sample_neighbors(g, frontiers[q], fanout)
            nbr = sg.edges()[0].long()
            src_q = torch.cat([frontiers[q], nbr]).unique()
            n.append(frontiers[q].numel()); e.append(nbr.numel()); src.append(src_q.numel())
            if l < k:                                 # dp layer: keep own frontier
                m.append(0)
                next_parts[q].append(src_q)
            else:                                     # split layer: scatter to owners
                own = owner[src_q]
                m.append(int((own != q).sum()))
                for p in range(G):
                    next_parts[p].append(src_q[own == p])
        rows.append(dict(layer=l, mode="dp" if l < k else "split",
                         n_max=max(n), n_sum=sum(n), e_max=max(e), e_sum=sum(e),
                         m_max=max(m), m_sum=sum(m), src_max=max(src),
                         u=torch.cat(frontiers).unique().numel()))
        frontiers = [torch.cat(parts).unique() for parts in next_parts]
    return rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dgl.seed(args.seed)
    fanouts = [int(f) for f in args.fanouts.split(",")]
    L, G, B = len(fanouts), args.world_size, args.batch_size
    if B % 2 or B % G:
        raise SystemExit(f"--batch_size must be divisible by 2 and by world_size ({G})")

    t0 = time.time()
    g, train_idx, owner = load(args)
    print(f"loaded {args.graph_name}: {g.num_nodes()} nodes, {g.num_edges()} edges "
          f"in {time.time() - t0:.1f}s")
    draw = SeedDrawer(train_idx, owner, G, B)
    # (system, k, seed function)
    strategies = ([("split", k, lambda: draw.split_loader(B)) for k in range(L + 1)]
                  + [("pipeline", 0, lambda: draw.split_loader(B // 2))]
                  + [("dgl", L, draw.dgl_loader)])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = ["graph_name", "model", "fanouts", "batch_size", "world_size", "partition_type", "seed",
            "system", "k", "minibatch", "layer", "mode", "n_max", "n_sum", "e_max", "e_sum",
            "m_max", "m_sum", "src_max", "u"]
    base = dict(graph_name=args.graph_name, model=args.model, fanouts=args.fanouts,
                batch_size=B, world_size=G, partition_type=args.partition_type, seed=args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for b in range(args.num_batches):
            for system, k, seed_fn in strategies:
                for row in run_system(g, owner, seed_fn(), fanouts, k, G):
                    w.writerow(dict(base, system=system, k=k, minibatch=b, **row))
            f.flush()
            print(f"minibatch {b + 1}/{args.num_batches} done ({time.time() - t0:.1f}s)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
