# Hybrid ablation run — decisions ledger

Autonomous run started 2026-11-03 ~20:15 PST (user out; "make reasonable calls and continue",
"keep track of key decisions"). This logs the key calls I made so they can be reviewed.

## Goal
Run the gsplit **hybrid** family across the Section-1 ablation cells to get ground-truth
"best" numbers (the cost-model/roofline simulator is on a separate branch and hard; get the
measured best first, then tune the simulator to match). One exploratory table:
`model | graph | layer(L) | h | hybrid type | total (training)`.

## Scope (confirmed with user before they left)
- **Interconnects:** both NVLink + PCIe (PCIe emulated via NCCL_P2P_DISABLE=1).
- **Graphs:** both papers100M + products.
- **Models:** both sage + gat.
- **Ablation cells (L,h):** the sec1 single-axis grid — {(2,1024),(3,256),(3,1024),(3,4096),(4,1024)}.
- **Hybrid matrix (user: "interior only"):**
  - DP hybrid `split --num_redundant_layer=k`, **k = 1 .. L-1** (k=0 full gsplit already in sec1; k=L pure DP skipped).
  - Pipeline hybrid `gsplit_pipepar --num_redundant_layer=k`, **k = 0 .. L-1** (k=0 = pure pipeline kept — it's the headline pipeline hybrid and per prior notes the likely best candidate; k=L degenerate).
- **Total: 200 cells** (100 papers / 100 products; 80 DP + 120 pipeline). Protocol matches sec1: 3 epochs, batch 1024, world_size 4, --random_feat, 0% cache, 1200 s/cell cap.

## Key decisions / reasonable calls
1. **Kept pipeline@k=0** despite "interior only" — my "interior" option's rationale for dropping k=0 was "gsplit already have it," which is only true for the *split* system. For pipeline, k=0 is pure pipeline (new, important), so I kept it. (DP split@k=0 and k=L are the ones dropped.)
2. **Fixed a real bug** in `gen_settings.py`: `identity_of`/`present_identities` did not include `num_redundant_layer`, so run_settings would have wrongly skipped k-variants of the same cell. Added k to the identity (backward-compatible: all sec1 rows are k=0). Without this the whole hybrid sweep would collapse to one k per cell.
3. **products provisioning via OGB** (ALLOW_OGB=1): products topology is not in B2 (only its w4 map is); OGB is deterministic so it aligns with the existing map. papers100M restores from B2 as before.
4. **Overnight safety (user unreachable):** run is fully self-terminating — a self-terminating wrapper (terminate on complete / provision-fail / MAX_HOURS) PLUS an independent idle-watchdog (terminate on 30-min idle or absolute max-life). Fail-toward-terminate so the $11/hr box can never idle-bill overnight. MAX_HOURS caps the heavy tail.
5. **3 epochs kept** (not reduced) so hybrid numbers are directly comparable to the sec1 gsplit@k=0 baseline.

## Timeline / status
- (filled in as the run progresses — see bottom)

## Artifacts
- Settings: `experiment/script/sec1_hybrid_settings.jsonl` (200 cells)
- Generator: `experiment/script/gen_hybrid_settings.py`
- Run wrapper: `experiment/script/cloud/mc/run_hybrid_autoterm.sh`
- Idle watchdog: `experiment/script/cloud/mc/idle_watchdog.sh`
- Table generator: `defense_latex/gen_hybrid_table.py`
- Raw results (pulled): `defense_latex/results/hybrid/`
- B2: `gsplit-data/groot/logs/sec1_autoterm/` (hybrid shard CSVs)

---
### Log
- 2026-11-03 ~20:15 PST: scope confirmed, 200-cell settings generated, identity bug fixed.
- 2026-11-03 20:30 PST (04:30 UTC 11-04): launched MC box uuid `545a9e81-3a46-4523-a408-a8cbb7ffa0a1`
  (IP 216.81.200.45, 8xA100, $11.04/hr), WARM volume attached. WARM B2 env restore ~2 min.
- 20:36 PST: run_hybrid_autoterm.sh (MAX_HOURS=12) + idle_watchdog.sh (IDLE_LIMIT=1800, MAXLIFE=12h) launched.
- 20:41 PST: VERIFIED running — both graphs provisioned (papers100M from B2; products regenerated
  from OGB cleanly), all 8 GPUs 75-83%, split@k=1 and split@k=2 running concurrently (confirms the
  identity fix — k-variants are no longer collapsed). 0 t/o / 0 oom so far. 200 cells queued.
- TODO on completion: box self-terminates -> pull hyb_shard{0,1}.csv from B2/VM, append to
  sweep_all.csv (additive), run defense_latex/gen_hybrid_table.py -> hybrid_ablation.{csv,tex}.
- WATCH: gsplit_pipepar (pipeline) cells are the less-tested path on papers100M/pcie; if they
  fail they record oom/timeout sentinels and the sweep continues (that is itself a result).
- 2026-11-04 ~00:45 PST: papers100M COMPLETE (100/100 hybrid cells). products still running
  (~20 cells left). Box 545a9e81 verified genuinely computing (GPUs 87-96%). Pulled papers
  hybrid rows, appended to sweep_all.csv (additive; +100 -> 368 rows), raw saved to
  defense_latex/results/hybrid/hyb_papers_shard{0,1}.csv. Backup: sweep_all.csv.bak_hyb_*.

### papers100M preliminary results (raw; user to verify)
- A hybrid beats baseline gsplit in ALL 18 papers100M cells (57 of the hybrid variants faster).
- DP hybrid (split, k>0): helps both SAGE+GAT; speedup grows with k; largest interior k
  (k=L-1) is fastest in 13/18 cells. Biggest on PCIe+deep/wide (SAGE PCIe L=4 DP k3: +33.8%).
- Pipeline hybrid (gsplit_pipepar): helps SAGE (+4..+20%); OOMs for GAT on ALL 30 cells.
- Table: defense_latex/tables/hybrid_papers100M.tex (input from sections/02_alternate_hybrids/
  hybrids.tex, new subsec:hybrid-empirical). Long CSV: defense_latex/results/hybrid_ablation.csv.
  Generator rewritten: defense_latex/gen_hybrid_table.py (per-graph, baseline|bestDP|bestpipe|Δ,
  epoch(train), interconnect rows; products table auto-emits once its rows land in sweep_all).

### Why GAT + pipeline OOMs (investigated 2026-11-04)
- STRUCTURAL, not a bug. The pipeline driver keeps TWO micro-batches fully live at once (both
  sampled + features fetched up front, both autograd graphs intact concurrently) -> ~2x peak
  activation memory vs single-batch split. (pipeline_split_trainer.py docstring Caveat 1 + Pipeline.forward.)
- GAT uses 4 attention heads (splitmodel.py GATConv), materializing per-edge attention +
  (N,heads,feat) activations. Baseline single-batch GAT already reserves up to 44.4 GB on
  papers100M (SAGE: 23.8 GB). 2x ~44 GB >> 40-48 GB A100 -> OOM on every GAT pipe cell,
  independent of (L,h,k) -- matching the uniform all-30 pattern. SAGE's 2x~24 GB mostly fits.
- Recorded as OOM sentinels and KEPT in place (no-data-removal rule); GAT pipeline simply not
  viable at these sizes without memory relief (e.g. not double-buffering features, or fewer heads).
- TODO products: on box completion, append ONLY graph==products rows from final shards (avoid
  dup of the 100 papers rows already in master), uncomment \input{tables/hybrid_products},
  re-run gen_hybrid_table.py, rebuild PDF.
