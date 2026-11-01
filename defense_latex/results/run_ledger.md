# Run ledger — dissertation evaluation (master memory)

Master = this session, executing each step inline (user chose inline-as-master,
authorized to go through the whole plan incl. GPU provisioning). Default order
1a → 1b → 1c → 3 → 4; Section 2 independent. One row per step: what it produced,
verdict, next decision.

---

## Preflight (master, 2026-10-31)
- `0cd49acb` (pipeline-split `gsplit_pipepar`) present in `git log` on
  `final_dissertation` ✓ — no merge needed.
- Clean start: no prior ledger, no `sweep_all.csv`, no `sweep.sh`, no
  `gen_results_table.py`. `main.sh` present (reference).
- Plan items already satisfied in-tree (no edit needed), noted so they aren't redone:
  - `papers100M` already in the real-feature list — `utils/dataloading.py:121`.
  - quiver PCIe replicated-cache already handled — `device_replicate` when
    `not nvlink` (`quiver_trainer.py`), so the "or it hangs" hazard is covered.

---

## Step 1a — plumbing (code only)  —  verdict: DONE (pending pod self-test)
Slave: master-inline. Produced (one reviewable diff):
- `utils/config.py` — added `hardware` (nvlink|pcie_emu) + `cache_frac` columns to
  `header()`/`content()`; `cache_frac` attr default -1.0.
- `utils/args.py` — new `--cache_frac` (float, fraction of feat; >0 overrides
  `--cache_size` for quiver).
- `train_main.py` — `nvlink=args.nvlink` (was hardcoded True; now the sweep's
  per-pass flag controls hardware label + quiver policy); wire `cfg.cache_frac`.
- `nodepred/quiver_trainer.py` — honor `cache_frac` (per-GPU device_replicate cache
  = frac × full feat); comment on why device_replicate is the PCIe-safe path.
- `experiment/script/sweep.sh` (new) — one hardware pass × one graph over
  {split k∈0..2, dgl, quiver@25%/10%, p3, gsplit_pipepar} × {sage,gat}; PCIe pass
  sets `NCCL_P2P_DISABLE=1` and drops `--nvlink`. Per-pass CSV `sweep_<g>_<hw>.csv`.
- `defense_latex/gen_results_table.py` (new) — CSV→`\input`-able LaTeX; fastest
  system per (graph,hardware,model,depth) cell highlighted `\rowcolor{red!15}`;
  `--self-test` renders a synthetic CSV (no real data).

Note: `gen_results_table.py --self-test` NOT yet run — CLAUDE.md forbids running
Python on WSL. Will run it on the pod as the first 1b action; fix-forward if needed.

Next decision: 1a reviewable-diff complete → commit, then proceed to 1b
(provision A100-SXM pod, validate generator self-test, run the sweep).

Follow-up fix (same step): `experiment/script/env.sh` — honor a preset
`WORKSPACE_DIR`/`data_dir` and use `realpath -m`, so the sweep runs on a RunPod
pod (`export WORKSPACE_DIR=/root`) instead of only the hardcoded Unity path.

---

## Step 1b — the sweep (compute)  —  verdict: IN PROGRESS
- Pod acquisition launched (bg): 4×A100-SXM-80GB, 250GB disk, `groot-eval-products`.
- Plan on pod: bootstrap_pod.sh (clone+build from GitHub) → sync_code.sh (push
  local tree incl. 1a, no rebuild: Python/bash only) → provision_data.sh products
  RESTORE_B2=1 → gen_results_table.py --self-test → smoke 1 cell → full sweep
  NVLink then PCIe (NCCL_P2P_DISABLE=1) → consolidate sweep_all.csv → commit.

## Section 2 — diagrams + cost-tradeoff (independent)  —  verdict: DONE (compile unverified)
Slave: master-inline (done during the pod bootstrap wait, since it needs no compute).
- `main.tex` — added `tikz` + libraries (positioning, arrows.meta, fit, backgrounds,
  calc, decorations.pathreplacing).
- `figures/data_split.tex` (new) — DP-then-split, L=4 k=2: blue DP layers (no shuffle)
  → switch/concat → red split layers (all-to-all per layer).
- `figures/pipeline_split.tex` (new) — two streams staggered one stage; A-conv ∥
  B-shuffle; epoch ~ max(compute,comm); one all-to-all in flight.
- `sections/02_alternate_hybrids/hybrids.tex` — wired both figures in; added the
  cost-tradeoff subsection (shuffle vs redundant-compute; NVLink cheap / PCIe-emu
  inflates shuffle; hidden size scales volume; the config challenge + k knob → motivates
  the simulator).
Caveat: LaTeX not compiled — CLAUDE.md forbids builds on WSL and the pod image has no
TeX. Needs a `latexmk main.tex` pass in the user's TeX env to confirm the figures render.

## Step 1b — the sweep (compute)  —  verdict: IN PROGRESS (pod building)
- Pod `u99vfp5yn57bgc`: 4×A100-SXM-80GB, 1001 GB RAM, 250 GB disk, $7.16/hr.
  SSH `root@154.54.102.52:10938` (key ~/.runpod/ssh/runpodctl-ssh-key).
- Bootstrap running detached on pod (nohup; survives SSH drops). Watcher bg task
  `bwateg3b3` notifies on DONE/stall. User confirmed this is an overnight job.
- Next (after bootstrap DONE): sync_code.sh (push local 1a/env.sh, no rebuild) →
  provision_data.sh products RESTORE_B2=1 → gen_results_table.py --self-test (on pod) →
  smoke 1 cell → full sweep NVLink then PCIe → scp sweep_*.csv back → consolidate →
  commit sweep_all.csv.

## Step 1a generator — validated (on pod, pandas)  —  verdict: DONE
- `gen_results_table.py --self-test` PASSES: valid tabular, red `\rowcolor` winner per
  cell, OOM renders, `25\%` single-escaped. Fixed self-test col/row misalignment (stray
  `depthx`), the double-%-escape, and the NaN-vs-OOM display (keep raw metric col, use a
  separate `__metric_num` for winner detection). main.tex: `\usepackage[table]{xcolor}`
  so `\rowcolor` compiles.
- Note: CLAUDE.md relaxed (user) to allow CPU-only data processing on WSL; but WSL python3
  lacks pandas and installs stay on RunPod, so the generator still runs on the pod for now.

## Step 1b environment — pod ready  —  verdict: DONE
- Bootstrap verified (torch 2.4+cu124, 4×A100, dgl imports). Local code synced.
- B2: `--with-creds` auto-copy FAILED silently (empty remote → "account not found");
  fixed by manual scp of b2.env + `rclone config create gsplit-b2`. Now authenticates.
- B2 layout: `groot/graph/{arxiv,papers100M}` + `groot/partition_map/{arxiv,papers100M}`
  (siblings). **products NOT in B2** → regenerate from OGB (provision fallback);
  papers100M IS in B2 (matches memory). products provisioning running (bg, pod).

## Section 3 — simulator writeup (step 1 only; 2/4 await 1b data)  —  verdict: PARTIAL
- simulator.tex scaffold was already accurate (compute measured per-layer NOT linear;
  PCIE_BW 16→25 documented; microprofile table with real numbers). Added the honesty
  sentence the plan requires: single-GPU estimate holds no optimizer/grad state, compute
  measured not linear. Validation rows (DGL baseline, papers100M fidelity) + predicted
  configs are DATA-DEPENDENT → blocked on 1b's sweep_all.csv. Not fabricating numbers.

## Master note — progress email channel (user request 2026-10-31)
User asked for periodic progress + blocker emails to sandeep06011991@gmail.com.
BLOCKER for that feature: no email channel available to the master —
- claude.ai Gmail connector needs OAuth, unavailable in this non-interactive session;
- no local mailer (msmtp/mail/sendmail) or SMTP creds on the control box.
Fallback until resolved: progress is logged here (committed) as the durable trail.
Options surfaced to user: (a) authorize the Gmail connector; (b) drop a Gmail
app password in ~/.config/groot/ so the POD can send via smtplib (pod may run
Python; WSL may not). The run itself is NOT blocked on this.
