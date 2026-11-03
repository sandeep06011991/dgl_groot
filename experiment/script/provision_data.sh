#!/usr/bin/env bash
# provision_data.sh — make a graph's run-time data available under $data_dir.
#
#   Usage:  provision_data.sh <arxiv|products|papers100M> ["world_sizes"]
#   e.g.    provision_data.sh papers100M "4"
#
# DEFAULT POLICY (do not special-case this for "overnight" — it IS the default):
#   * Backblaze B2 is the SOURCE OF TRUTH. By default we RESTORE from B2:
#       - the METIS partition map(s)  (expensive + non-reproducible -> never rebuilt by default)
#       - the graph TOPOLOGY + idx splits (indptr/indices, train/valid/test_idx)
#   * We do NOT touch OGB and do NOT run METIS by default. Both are explicit opt-ins.
#   * FEATURES/LABELS are fetched ONLY when a run actually needs real features
#     (NEED_FEAT=1). Section-1 perf runs use --random_feat, which generates feat AND
#     label at runtime (see utils/dataloading.py), so feat.npy/label.npy are never
#     needed -> the giant (e.g. ~57GB papers100M) feature tensor is never pulled/built.
#   * If something required is missing and its opt-in is off, we HARD-FAIL loudly
#     instead of silently falling back to a ~70GB OGB download or a segfault-prone
#     METIS run. (Overnight unattended runs must never block or burn hours this way.)
#
# Env knobs (defaults chosen for the policy above):
#   data_dir     where graphs live                         (default /root/graph)
#   RESTORE_B2   restore from B2 first                      (DEFAULT 1)
#   ALLOW_OGB    permit regenerating topology/idx/feat from OGB on a B2 miss (DEFAULT 0)
#   ALLOW_METIS  permit (re)building the partition map with METIS on a B2 miss (DEFAULT 0)
#   NEED_FEAT    this provision is for REAL-feature runs -> ensure feat.npy+label.npy
#                                                            (DEFAULT 0: random_feat runs)
#   B2_REMOTE    rclone remote name                         (default gsplit-b2)
#   B2_PREFIX    path inside the bucket                     (default gsplit-data/groot)
#   WEIGHT_WS    #GPUs for freq-weight step (METIS only)    (default 1)
#   PART_NODE/PART_EDGE/PART_BAL  partition scheme          (default dst/freq/xbal ->
#                map token ndst_efreq_xbal; MUST match utils/args.py training defaults)
#   PUSH_MAPS    push partition maps to B2                  (DEFAULT 1: small + non-reproducible)
#   PUSH_B2      also push the graph topology/idx dir to B2 (DEFAULT 0); feat.npy/label.npy
#                are EXCLUDED from that push unless PUSH_FEAT=1 (keeps B2 under its cap)
#   PUSH_FEAT    include feat.npy/label.npy in a PUSH_B2 push (DEFAULT 0)
#
# Produces, under $data_dir:
#   <graph>/{indptr_xsym,indices_xsym,train_idx,valid_idx,test_idx}.npy  (+feat/label if NEED_FEAT)
#   partition_map/<graph>/<graph>_w<k>_ndst_efreq_xbal.npy   for each world size
set -uo pipefail

GRAPH="${1:?usage: provision_data.sh <graph> [world_sizes]}"
WORLD_SIZES="${2:-2 4}"
: "${data_dir:=/root/graph}"
: "${RESTORE_B2:=1}"
: "${ALLOW_OGB:=0}"
: "${ALLOW_METIS:=0}"
: "${NEED_FEAT:=0}"
: "${B2_REMOTE:=gsplit-b2}"
: "${B2_PREFIX:=gsplit-data/groot}"
: "${WEIGHT_WS:=1}"
: "${PART_NODE:=dst}"
: "${PART_EDGE:=freq}"
: "${PART_BAL:=xbal}"
: "${PUSH_MAPS:=1}"
: "${PUSH_B2:=0}"
: "${PUSH_FEAT:=0}"
# Partition-scheme token the map filename carries; the SAME string training builds
# in load_partition_map (get_partition_type -> n<node>_e<edge>_<bal>).
PTYPE="n${PART_NODE}_e${PART_EDGE}_${PART_BAL}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PD="$REPO/experiment"
export PYTHONPATH="$PD:${PYTHONPATH:-}"

# ensure the conda env is active (often run in a fresh SSH shell)
: "${ENV_PATH:=/home/groot_env}"
if [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_PATH" 2>/dev/null || true
fi
GDIR="$data_dir/$GRAPH"
MDIR="$data_dir/partition_map/$GRAPH"
mkdir -p "$GDIR" "$MDIR"

case "$GRAPH" in
  arxiv)      ZIP=arxiv.zip;          RAW=arxiv;          SYM=xsym ;;
  products)   ZIP=products.zip;       RAW=products;       SYM=xsym ;;
  papers100M) ZIP=papers100M-bin.zip; RAW=papers100M-bin; SYM=xsym ;;
  *) echo "unsupported graph '$GRAPH' (this script covers the OGB graphs)"; exit 2 ;;
esac
OGB_BASE="http://snap.stanford.edu/ogb/data/nodeproppred"

have_b2(){ command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^${B2_REMOTE}:"; }
topo_ok(){ [ -f "$GDIR/indptr_${SYM}.npy" ] && [ -f "$GDIR/indices_${SYM}.npy" ] \
           && [ -f "$GDIR/train_idx.npy" ] && [ -f "$GDIR/valid_idx.npy" ] && [ -f "$GDIR/test_idx.npy" ]; }

# regenerate topology+idx(+feat+label) from OGB. Only reached via ALLOW_OGB / NEED_FEAT.
ogb_regen(){
  echo "-- regenerating from OGB (download ~zip + get_npgraph) --"
  ( cd "$data_dir"
    if [ ! -d "ogbn_${GRAPH}" ]; then
      echo "   downloading $ZIP ..."
      wget -q "$OGB_BASE/$ZIP"
      unzip -q "$ZIP"
      mv -f "$RAW" "ogbn_${GRAPH}"
      rm -f "$ZIP"
    fi )
  python3 "$PD/prepare_dataset/get_npgraph.py" --data_dir="$data_dir" --graph_name="$GRAPH"
}

echo "== provision $GRAPH -> $data_dir  (world_sizes: $WORLD_SIZES, scheme: $PTYPE) =="
echo "   policy: RESTORE_B2=$RESTORE_B2 ALLOW_OGB=$ALLOW_OGB ALLOW_METIS=$ALLOW_METIS NEED_FEAT=$NEED_FEAT"

# ---- 1. restore from B2 (DEFAULT source of truth) ----------------------------
if [ "$RESTORE_B2" = 1 ]; then
  if have_b2; then
    echo "-- restoring from B2 ($B2_REMOTE:$B2_PREFIX) --"
    EXCL=()
    [ "$NEED_FEAT" = 1 ] || EXCL=(--exclude "feat.npy" --exclude "label.npy")  # skip the giant feature tensor
    rclone copy "$B2_REMOTE:$B2_PREFIX/graph/$GRAPH" "$GDIR" -P "${EXCL[@]}" || echo "WARN B2 graph restore incomplete"
    rclone copy "$B2_REMOTE:$B2_PREFIX/partition_map/$GRAPH" "$MDIR" -P || echo "WARN B2 map restore incomplete"
  else
    echo "WARN RESTORE_B2=1 but no '$B2_REMOTE' rclone remote — set up B2 creds per"
    echo "     RUNBOOK §Prerequisites (sync_code.sh --with-creds). Cannot restore from B2."
  fi
fi

# ---- 2. TOPOLOGY + idx (needed by EVERY run) ---------------------------------
if topo_ok; then
  echo "-- topology + idx present --"
elif [ "$ALLOW_OGB" = 1 ]; then
  ogb_regen
  topo_ok || { echo "ERROR: topology/idx still missing after OGB regen"; exit 3; }
else
  echo "ERROR: topology/idx for '$GRAPH' are not on B2 and ALLOW_OGB=0."
  echo "       Refusing to silently download ~70GB from OGB. Either:"
  echo "         - push the prepared topology+idx to B2 (PUSH_B2=1), or"
  echo "         - re-run with ALLOW_OGB=1 to regenerate from OGB."
  exit 3
fi

# ---- 3. FEATURES/LABELS — only when real features are needed -----------------
if [ "$NEED_FEAT" = 1 ]; then
  if [ -f "$GDIR/feat.npy" ] && [ -f "$GDIR/label.npy" ]; then
    echo "-- feat/label present (NEED_FEAT=1) --"
  elif [ "$ALLOW_OGB" = 1 ]; then
    echo "-- NEED_FEAT=1 and feat/label missing: regenerate from OGB --"
    ogb_regen
    { [ -f "$GDIR/feat.npy" ] && [ -f "$GDIR/label.npy" ]; } || { echo "ERROR: feat/label still missing"; exit 3; }
  else
    echo "ERROR: NEED_FEAT=1 but feat/label missing from B2 and ALLOW_OGB=0."
    echo "       Set ALLOW_OGB=1 to pull features from OGB (large), or store them on B2."
    exit 3
  fi
else
  echo "-- NEED_FEAT=0: skipping feat/label (runtime --random_feat generates both) --"
fi

# ---- 4. PARTITION MAPS — from B2 by default; METIS only via ALLOW_METIS ------
for k in $WORLD_SIZES; do
  MAP="$MDIR/${GRAPH}_w${k}_${PTYPE}.npy"
  if [ -f "$MAP" ]; then echo "map w$k present (from B2)"; continue; fi
  if [ "$ALLOW_METIS" != 1 ]; then
    echo "ERROR: partition map w$k is not on B2 and ALLOW_METIS=0:"
    echo "       $MAP"
    echo "       METIS regeneration is OFF by default (slow + segfault-prone on huge"
    echo "       graphs). Set ALLOW_METIS=1 to build it, or push the map to B2."
    exit 3
  fi
  echo "-- ALLOW_METIS=1: building partition map w$k (METIS, scheme=$PTYPE) --"
  # freq weights are only needed to build maps
  if [ ! -f "$GDIR/node_weight.npy" ] || [ ! -f "$GDIR/edge_weight.npy" ]; then
    echo "   freq weights missing: get_weight_fast (fanout 15,15,15, ws=$WEIGHT_WS)"
    # NOTE: get_weight_fast.py calls os._exit(1) even on success; gate on the output files.
    python3 "$PD/prepare_dataset/get_weight_fast.py" --data_dir="$data_dir" --graph_name="$GRAPH" \
      --fanouts=15,15,15 --num_epoch=5 --world_size="$WEIGHT_WS" || true
  fi
  # METIS can segfault nondeterministically on huge graphs; retry once.
  for attempt in 1 2; do
    python3 "$PD/prepare_dataset/get_partition.py" --data_dir="$data_dir" --graph_name="$GRAPH" \
      --num_partition="$k" --node_weight="$PART_NODE" --edge_weight="$PART_EDGE" --bal="$PART_BAL" \
      || echo "WARN get_partition k=$k attempt $attempt crashed"
    [ -f "$MAP" ] && break
    echo "   retrying map w$k ..."
  done
  [ -f "$MAP" ] || { echo "ERROR: partition map w$k NOT produced by METIS"; exit 3; }
done

# ---- 5. push to B2 so the next run can restore (keeps B2 the source of truth) ----
if { [ "$PUSH_MAPS" = 1 ] || [ "$PUSH_B2" = 1 ]; } && ! have_b2; then
  echo "WARN B2 push requested but no '$B2_REMOTE' rclone remote — skipping push."
else
  if [ "$PUSH_MAPS" = 1 ] && have_b2; then
    echo "-- pushing partition maps to B2 --"
    rclone copy "$MDIR" "$B2_REMOTE:$B2_PREFIX/partition_map/$GRAPH" -P || echo "WARN map push failed"
  fi
  if [ "$PUSH_B2" = 1 ] && have_b2; then
    PEXCL=()
    [ "$PUSH_FEAT" = 1 ] || PEXCL=(--exclude "feat.npy" --exclude "label.npy")  # don't blow the B2 cap with features
    echo "-- PUSH_B2=1: pushing graph/$GRAPH topology+idx to B2 (PUSH_FEAT=$PUSH_FEAT) --"
    rclone copy "$GDIR" "$B2_REMOTE:$B2_PREFIX/graph/$GRAPH" -P --transfers=16 --b2-chunk-size=100M "${PEXCL[@]}" \
      || echo "WARN graph push failed (B2 storage cap?)"
  fi
fi

echo "== provision $GRAPH done =="
echo "graph dir:"; ls -la "$GDIR"
echo "maps:";      ls -la "$MDIR"
