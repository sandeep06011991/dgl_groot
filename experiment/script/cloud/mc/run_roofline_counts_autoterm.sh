#!/usr/bin/env bash
# run_roofline_counts_autoterm.sh — RUN ON THE VM (Massed Compute, CPU box is enough).
# Unattended collection of sampling-only roofline counts (experiment/roofline/sample_counts.py)
# for every graph/model/depth of the sweep -> push each CSV to B2 -> SELF-TERMINATE.
#
# No GPU and no groot build: stock CPU torch + DGL wheels. Data is B2-restore first
# (papers100M topology + w4 maps); products topology is rebuilt from OGB because it is
# missing in B2 (features skipped) and pushed back to B2.
#
# Required env (from ~/roofline_job.env, scp'd before start):
#   MC_API_TOKEN, SELF_UUID        to self-terminate
#   ~/.config/groot/b2.env         B2_KEY_ID / B2_APP_KEY
# Optional: MAX_HOURS (6), NO_TERMINATE=1 (debug)
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
W="$HOME/roofline"; DATA="$W/data"; OUT="$W/out"; VENV="$W/venv"
MAX_HOURS="${MAX_HOURS:-6}"
B2_LOGS="gsplit-b2:gsplit-data/groot/logs/roofline_counts_v2"
mkdir -p "$DATA" "$OUT"
exec > >(tee -a "$W/job.out") 2>&1
log() { echo "[roofline $(date -u +%FT%TZ)] $*"; }
log "START self=${SELF_UUID:-UNSET} max_hours=$MAX_HOURS"

push_out() {
  rclone copy "$OUT" "$B2_LOGS/" 2>&1 | tail -3
  rclone copy "$W/job.out" "$B2_LOGS/" 2>&1 | tail -1
}
terminate_self() {
  log "TERMINATE ($1)"
  push_out || log "WARN push failed"
  [ -n "${NO_TERMINATE:-}" ] && { log "NO_TERMINATE set; staying up"; return 0; }
  if [ -z "${MC_API_TOKEN:-}" ] || [ -z "${SELF_UUID:-}" ]; then
    log "cannot API-terminate; poweroff fallback"; sudo poweroff; return 1
  fi
  curl -s --max-time 30 -X POST -H "Authorization: Bearer $MC_API_TOKEN" -H "Content-Type: application/json" \
    -d "{\"instanceUuids\":[\"$SELF_UUID\"]}" "$API/instance/terminate" -w $'\nHTTP %{http_code}\n'
  sleep 120; sudo poweroff   # belt and braces if the API call did not take
}
fail() { terminate_self "fail: $1"; exit 1; }
( sleep "$((MAX_HOURS*3600))"; log "MAX_HOURS cap hit"; terminate_self "max-hours" ) &

# ---- 1) toolchain ---------------------------------------------------------------
log "apt + rclone"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null && \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip unzip curl >/dev/null \
  || fail "apt"
command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || fail "rclone install"
# shellcheck disable=SC1091
source "$HOME/.config/groot/b2.env" || fail "no b2.env"
rclone config create gsplit-b2 b2 account "$B2_KEY_ID" key "$B2_APP_KEY" >/dev/null || fail "rclone config"
rclone lsd gsplit-b2:gsplit-data/groot >/dev/null || fail "B2 not reachable"

log "python venv: CPU torch + matching DGL (pinned; PyPI 2.1 first, then 2.4 repo)"
python3 -m venv "$VENV" && source "$VENV/bin/activate" || fail "venv"
pip install -q --upgrade pip
check_dgl() { python -c "
import torch, dgl
g = dgl.graph(([0, 1, 2], [1, 2, 0]))
assert dgl.sampling.sample_neighbors(g, torch.tensor([0, 1]), 2).num_edges() == 2
import ogb, sklearn, pandas                         # everything the products rebuild imports
from ogb.nodeproppred import DglNodePropPredDataset
print('torch', torch.__version__, 'dgl', dgl.__version__, 'ogb', ogb.__version__,
      'threads', torch.get_num_threads())"; }
install_combo() {  # $1 torch  $2 dgl  $3 torchdata  $4 extra pip args for dgl
  pip uninstall -y -q dgl torch torchdata >/dev/null 2>&1
  # full dependency resolution, but torch and numpy can never be swapped out
  printf 'torch==%s\nnumpy<2\n' "$1" > "$W/constraints.txt"
  pip install -q "torch==$1" --index-url https://download.pytorch.org/whl/cpu &&
  pip install -q -c "$W/constraints.txt" "numpy<2" scipy pandas pyyaml pydantic \
      "torchdata==$3" ogb &&
  pip install -q -c "$W/constraints.txt" "dgl==$2" $4 &&
  check_dgl
}
# data.dgl.ai (DGL's own wheel host) failed SSL on 2026-11-15, so PyPI's dgl 2.1.0 first.
install_combo 2.1.0 2.1.0 0.7.0 "" \
  || { log "combo torch2.1/dgl2.1.0 (PyPI) failed; trying torch2.4/dgl2.4.0 (data.dgl.ai)";
       install_combo 2.4.0 2.4.0 0.8.0 "-f https://data.dgl.ai/wheels/torch-2.4/repo.html"; } \
  || fail "import dgl"

# ---- 2) data --------------------------------------------------------------------
log "B2 restore: papers100M topology + idx, w4 maps"
mkdir -p "$DATA/papers100M" "$DATA/partition_map"
rclone copy gsplit-b2:gsplit-data/groot/graph/papers100M "$DATA/papers100M" --transfers=8 \
  --include "indptr_xsym.npy" --include "indices_xsym.npy" --include "*_idx.npy" || fail "papers restore"
for g in products papers100M; do
  rclone copy "gsplit-b2:gsplit-data/groot/partition_map/$g" "$DATA/partition_map/$g" \
    --include "${g}_w4_ndst_efreq_xbal.npy" || fail "map restore $g"
done

log "products topology: B2 miss -> rebuild from OGB (no features), push back to B2"
python - "$DATA" <<'EOF' || fail "products OGB rebuild"
import os, sys, numpy as np
import ogb.nodeproppred.dataset_dgl as ogb_dgl
ogb_dgl.decide_download = lambda url: True        # OGB asks y/N for >1GB; unattended job
from ogb.nodeproppred import DglNodePropPredDataset
data = sys.argv[1]; out = os.path.join(data, "products"); os.makedirs(out, exist_ok=True)
ds = DglNodePropPredDataset("ogbn-products", os.path.join(data, "ogb"))
g = ds[0][0]
indptr, indices, _ = g.adj_tensors("csc")          # same steps as prepare_dataset/get_npgraph.py
np.save(f"{out}/indptr_xsym.npy", indptr.numpy()); np.save(f"{out}/indices_xsym.npy", indices.numpy())
for k, v in ds.get_idx_split().items():
    np.save(f"{out}/{'valid' if k == 'valid' else k}_idx.npy", v.numpy())
pm = np.load(f"{data}/partition_map/products/products_w4_ndst_efreq_xbal.npy")
assert pm.shape[0] == indptr.shape[0] - 1, (pm.shape, indptr.shape)
print("products ok:", indptr.shape[0] - 1, "nodes", indices.shape[0], "edges")
EOF
rclone copy "$DATA/products" gsplit-b2:gsplit-data/groot/graph/products/ --include "*.npy" \
  || log "WARN products push to B2 failed (non-fatal)"
for g in products papers100M; do
  ls "$DATA/$g/indptr_xsym.npy" "$DATA/$g/indices_xsym.npy" "$DATA/$g/train_idx.npy" \
     "$DATA/partition_map/$g/${g}_w4_ndst_efreq_xbal.npy" >/dev/null || fail "missing input for $g"
done

# ---- 3) counts --------------------------------------------------------------------
# graph model fanouts num_minibatches   (cheap first; papers 6-layer last, few batches)
CONFIGS="
products sage 15,15 20
products gat 15,15 20
products sage 15,15,15 20
products gat 15,15,15 20
products sage 15,15,15,15 20
products gat 15,15,15,15 20
products sage 15,15,15,15,15,15 5
products gat 15,15,15,15,15,15 5
papers100M sage 15,15 20
papers100M gat 15,15 20
papers100M sage 15,15,15 20
papers100M gat 15,15,15 20
papers100M sage 15,15,15,15 10
papers100M gat 15,15,15,15 10
papers100M sage 15,15,15,15,15,15 2
"
n_ok=0; n_fail=0
while read -r g m f nb; do
  [ -z "$g" ] && continue
  L=$(echo "$f" | tr ',' '\n' | wc -l)
  tag="${g}_${m}_${L}x15_b1024"
  log "counts $tag (minibatches=$nb)"
  if python "$W/sample_counts.py" --data_dir="$DATA" --graph_name="$g" --model="$m" \
       --fanouts="$f" --batch_size=1024 --world_size=4 --num_batches="$nb" \
       --out="$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1; then
    n_ok=$((n_ok+1)); log "done $tag ($(wc -l < "$OUT/$tag.csv") rows)"
  else
    n_fail=$((n_fail+1)); log "FAILED $tag: $(tail -2 "$OUT/$tag.log" | tr '\n' ' ')"
  fi
  push_out
done <<< "$CONFIGS"

log "ALL DONE ok=$n_ok failed=$n_fail"
touch "$OUT/DONE"
terminate_self "complete"
