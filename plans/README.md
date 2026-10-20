# Novel Parallelism Mechanisms for gsplit — Plans

Two new parallelism schemes to build on top of the existing Spara/gsplit **split**
(partition-aware) system:

1. **Data-parallelism + Split** — [`01_dp_plus_split.md`](01_dp_plus_split.md)
   Run the first *k* GNN layers replicated (pure data-parallel, no cross-GPU shuffle)
   and switch to split (partition-aware shuffle) only from layer *k* onward. Expected
   to be the easier one: gsplit already has the **shuffle** mechanism; the idea is to
   *trigger the shuffle at a later layer* rather than from layer 0.

2. **Pipeline + Split** — [`02_pipeline_plus_split.md`](02_pipeline_plus_split.md)
   Two micro-batch **streams** run alternately on the *same* model; the backward
   passes accumulate gradients before a single optimizer step. Expected to be mostly an
   **external driver** change (not a model change).

## Methodology (applies to both)

**Do NOT prototype against the full GNN model.** Instead, build **micro-tests that
capture the intent at a small scale**, so correctness is provable analytically and
decoupled from GNN/DGL complexity:

- **Toy graph: an n-clique** (complete graph `K_n`, small `n`, e.g. 8–64 nodes).
  Every node neighbors every other node, so under any partition map *every* node has
  cross-partition neighbors — the maximal-communication stress case for the shuffle,
  and its structure is trivial to reason about.
- **Toy "model":** a trivial, analytically-checkable op per layer (e.g. sum/mean
  aggregation + a fixed or identity linear map), NOT SAGE/GAT. The point is to verify
  the *parallelism mechanism* (what data moves, when the shuffle fires, how gradients
  accumulate), not GNN accuracy.
- **Correctness oracle:** a single-GPU / single-stream reference computation on the
  same n-clique + toy op. The parallel mechanism must produce **bitwise-or-tolerance
  equal** forward outputs and **equal gradients** to the reference. That is the pass
  criterion for each micro-test.
- **Scale:** micro-tests run on **2 GPUs** (and conceptually any world size) with a
  tiny graph, so they run on a **cheap GPU** — no A100-SXM needed.

## Workflow

- Branch: **`final_dissertation_other_pipelines`** (all work here).
- Phase 1 (now): write these plans; no implementation.
- Phase 2 (after review): user kicks off a build cycle on a **cheaper GPU**; implement
  the micro-tests first, prove correctness on the n-clique, then graduate to the real
  model.

Each plan file contains: the mechanism, what in the existing code it reuses/changes,
the micro-test design (n-clique + toy op + oracle), the correctness checks, and open
questions/risks.
