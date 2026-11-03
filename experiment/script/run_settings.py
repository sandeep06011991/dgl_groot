#!/usr/bin/env python3
"""run_settings.py -- iterate the settings file produced by gen_settings.py and run
each cell, appending results to a log CSV. Idempotent and crash-safe:

  * each cell runs ``train_main.py`` under ``timeout -k 30 <cap>`` (default 1200s);
  * rc in {124,137}  -> a ``timeout`` sentinel row is recorded;
  * any other nonzero -> an ``oom`` sentinel row is recorded (trainers os._exit on OOM
    and write no row of their own, so without this the cell would be retried forever);
  * orphaned DDP workers are pkill'd between cells so a dead run can't poison the next;
  * before each cell the runner re-checks the master CSV *and* its own log file and skips
    any cell already present, so interrupting and re-running resumes cleanly.

Interconnect integrity: ``interconnect`` in each settings row drives BOTH switches together
(nvlink -> ``--nvlink`` + NCCL P2P on; pcie -> drop ``--nvlink`` + ``NCCL_P2P_DISABLE=1``),
so the recorded ``hardware`` column always matches what actually ran.

Usage (on the pod, inside the conda env):
    python3 run_settings.py --data_dir "$data_dir"
    python3 run_settings.py --dry-run          # print the commands, run nothing
    python3 run_settings.py --timeout 1200 --log_file sec1_runs.csv
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
EXP_DIR = os.path.normpath(os.path.join(HERE, ".."))          # experiment/
REPO = os.path.normpath(os.path.join(EXP_DIR, ".."))           # repo root
BASE_PYPATH = os.path.join(REPO, "python")
DEFAULT_SETTINGS = os.path.join(HERE, "sec1_settings.jsonl")
DEFAULT_MASTER = os.path.normpath(os.path.join(REPO, "defense_latex", "results", "sweep_all.csv"))

sys.path.insert(0, EXP_DIR)
import gen_settings as gs  # noqa: E402  (same-dir import; reuse grid/identity logic)


def load_settings(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def log_path_for(log_file):
    return os.path.join(EXP_DIR, "logs", log_file)


def present_from(paths):
    """Union of present 0%-cache identities across one or more CSVs (missing files ignored)."""
    ids = set()
    for p in paths:
        ids |= gs.present_identities(p)
    return ids


def build_argv(cell, data_dir):
    """The train_main.py arg vector for one settings row (interconnect -> --nvlink handled by caller)."""
    A = [
        f"--system={cell['system']}",
        f"--model={cell['model']}",
        f"--fanouts={cell['fanout']}",
        f"--hid_size={cell['hid_size']}",
        f"--graph_name={cell['graph']}",
        f"--data_dir={data_dir}",
        f"--batch_size={cell['batch_size']}",
        f"--world_size={cell['world_size']}",
        f"--num_epoch={cell['num_epoch']}",
        f"--num_redundant_layer={cell['num_redundant_layer']}",
        f"--cache_size={cell['cache_size']}",
    ]
    if float(cell.get("cache_frac", -1.0)) > 0:
        A.append(f"--cache_frac={cell['cache_frac']}")
    if cell.get("random_feat"):
        A.append("--random_feat")
    if cell["interconnect"] == "nvlink":
        A.append("--nvlink")
    return A


def cell_env(cell):
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_PYPATH + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if cell["interconnect"] == "nvlink":
        env.pop("NCCL_P2P_DISABLE", None)
    else:
        env["NCCL_P2P_DISABLE"] = "1"   # emulate PCIe: force NCCL comm off the NVLink mesh
    return env


def cleanup():
    """Kill orphaned DDP workers from the previous cell (bracket patterns avoid self-match)."""
    for pat in ("[t]rain_main.py", "[m]ultiprocessing.spawn"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    subprocess.run(["sleep", "3"], check=False)


def record_sentinel(cell, data_dir, kind, log_file):
    """Write a single 'timeout'/'oom' sentinel row to the log CSV (same Config schema, no run)."""
    from utils import get_partition_type, Config            # noqa: E402
    from utils.profiler import oom_profiler, timeout_profiler
    from utils.utils import write_to_csv
    prof = timeout_profiler() if kind == "timeout" else oom_profiler()
    cfg = Config(
        graph_name=cell["graph"], world_size=cell["world_size"], num_partition=cell["world_size"],
        num_epoch=cell["num_epoch"], fanouts=[int(x) for x in cell["fanout"].split(",")],
        batch_size=cell["batch_size"], system=cell["system"], model=cell["model"],
        cache_size=cell["cache_size"], hid_size=cell["hid_size"],
        log_path=log_path_for(log_file), data_dir=data_dir,
        nvlink=(cell["interconnect"] == "nvlink"),
        partition_type=get_partition_type("dst", "freq", "xbal"), sample_mode="uva",
    )
    cfg.num_redundant_layer = cell["num_redundant_layer"]
    cfg.cache_frac = float(cell.get("cache_frac", -1.0))
    cfg.random_feat = bool(cell.get("random_feat"))
    write_to_csv(log_path_for(log_file), [cfg], [prof])
    print(f"    recorded {kind.upper()} sentinel row -> logs/{log_file}")


def main():
    ap = argparse.ArgumentParser(description="run the Section-1 settings file")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS)
    ap.add_argument("--data_dir", default=os.environ.get("data_dir", ""), help="graph data dir (or $data_dir)")
    ap.add_argument("--log_file", default="sec1_runs.csv", help="output under experiment/logs/")
    ap.add_argument("--master", default=DEFAULT_MASTER, help="master CSV also consulted for skip")
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell wall-clock cap (s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run at most N cells (0 = all)")
    args = ap.parse_args()

    if not args.data_dir and not args.dry_run:
        sys.exit("error: --data_dir is required (or export $data_dir); see experiment/script/env.sh")

    cells = load_settings(args.settings)
    train_main = os.path.join(EXP_DIR, "train_main.py")
    log_csv = log_path_for(args.log_file)
    ran = skipped = 0

    for i, cell in enumerate(cells, 1):
        present = present_from([args.master, log_csv])  # re-read each cell: crash-safe resume
        if gs.identity_of(cell) in present:
            skipped += 1
            continue
        if args.limit and ran >= args.limit:
            print(f"--limit {args.limit} reached; stopping.")
            break

        A = build_argv(cell, args.data_dir or "$data_dir")
        A.append(f"--log_file={args.log_file}")
        tag = (f"{cell['graph']}/{cell['hardware']}/{cell['system']}/{cell['model']} "
               f"L={cell['depth']} h={cell['hid_size']}")
        cmd = ["timeout", "-k", "30", str(args.timeout), "python3", train_main] + A
        print(f">>> [{i}/{len(cells)}] {tag}  (cap {args.timeout}s, NCCL_P2P_DISABLE="
              f"{'1' if cell['interconnect'] != 'nvlink' else 'unset'})")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        rc = subprocess.run(cmd, env=cell_env(cell)).returncode
        if rc in (124, 137):
            print(f"!!! TIMEOUT (rc={rc})")
            cleanup()
            record_sentinel(cell, args.data_dir, "timeout", args.log_file)
        elif rc != 0:
            print(f"!!! NONZERO rc={rc} (treating as OOM/crash)")
            cleanup()
            record_sentinel(cell, args.data_dir, "oom", args.log_file)
        else:
            cleanup()
        ran += 1

    print(f"\ndone: {ran} run, {skipped} skipped (already present) -> logs/{args.log_file}")


if __name__ == "__main__":
    main()
