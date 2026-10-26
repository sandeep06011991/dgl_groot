#!/usr/bin/env bash
# provision_data.sh — make a graph's run-time data available under $data_dir,
# restoring from Backblaze B2 when possible, otherwise regenerating from OGB.
# Idempotent: only (re)builds what is missing. Safe to re-run.
#
#   Usage:  provision_data.sh <arxiv|products|papers100M> ["world_sizes"]
#   e.g.    provision_data.sh papers100M "2 4"
#
# Default path is REGENERATE-FROM-OGB (fast + deterministic). B2 restore is opt-in.
#
# Env knobs:
#   data_dir    where graphs live            (default /root/graph)
#   RESTORE_B2  1 = try restoring from B2 first (default 0 = go straight to OGB)
#   B2_REMOTE   rclone remote name           (default gsplit-b2)
#   B2_PREFIX   path inside the bucket       (default gsplit-data/groot)
#   WEIGHT_WS   #GPUs for freq-weight step   (default 1)
#   PART_NODE/PART_EDGE/PART_BAL  partition scheme (default dst/freq/xbal ->
#               map name n<node>_e<edge>_<bal> = ndst_efreq_xbal; MUST match the
#               training defaults in utils/args.py or load_partition_map won't find it)
#   PUSH_MAPS   1 = push partition maps to B2 (default 1: maps are small +
#               non-reproducible = Tier A2; set 0 to skip)
#   PUSH_B2     1 = ALSO push the large graph/<g>/ dir to B2 (default 0: it
#               regenerates from OGB and can hit the B2 storage cap)
#
# Produces, under $data_dir:
#   <graph>/{indptr_xsym,indices_xsym,train_idx,valid_idx,test_idx,feat,label}.npy
#   partition_map/<graph>/<graph>_w<k>_ndst_efreq_xbal.npy   for each world size
set -uo pipefail

GRAPH="${1:?usage: provision_data.sh <graph> [world_sizes]}"
WORLD_SIZES="${2:-2 4}"
: "${data_dir:=/root/graph}"
: "${RESTORE_B2:=0}"
: "${B2_REMOTE:=gsplit-b2}"
: "${B2_PREFIX:=gsplit-data/groot}"
: "${WEIGHT_WS:=1}"
: "${PART_NODE:=dst}"
: "${PART_EDGE:=freq}"
: "${PART_BAL:=xbal}"
: "${PUSH_MAPS:=1}"
: "${PUSH_B2:=0}"
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

echo "== provision $GRAPH -> $data_dir (world_sizes: $WORLD_SIZES, scheme: $PTYPE) =="

# ---- 1. (opt-in) restore from B2; default is to regenerate from OGB ----
if [ "$RESTORE_B2" = 1 ]; then
  if have_b2; then
    echo "-- RESTORE_B2=1: restoring from B2 ($B2_REMOTE:$B2_PREFIX) --"
    rclone copy "$B2_REMOTE:$B2_PREFIX/graph/$GRAPH" "$GDIR" -P || true
    rclone copy "$B2_REMOTE:$B2_PREFIX/partition_map/$GRAPH" "$MDIR" -P || true
  else
    echo "WARN RESTORE_B2=1 but no '$B2_REMOTE' rclone remote — set up B2 creds per"
    echo "     RUNBOOK §Prerequisites (or sync_code.sh --with-creds). Falling back to OGB."
  fi
else
  echo "-- regenerating from OGB (RESTORE_B2=$RESTORE_B2) --"
fi

# ---- 2. regenerate from OGB whatever is still missing ----
# get_npgraph writes topology + idx + feat + label together (deterministic from OGB),
# so run it if ANY of those is missing.
if [ ! -f "$GDIR/indptr_${SYM}.npy" ] || [ ! -f "$GDIR/train_idx.npy" ] || [ ! -f "$GDIR/feat.npy" ]; then
  echo "-- (topology/idx/feat) missing: download + convert from OGB --"
  ( cd "$data_dir"
    if [ ! -d "ogbn_${GRAPH}" ]; then
      echo "   downloading $ZIP ..."
      wget -q "$OGB_BASE/$ZIP"
      unzip -q "$ZIP"
      mv -f "$RAW" "ogbn_${GRAPH}"
      rm -f "$ZIP"
    fi )
  python3 "$PD/prepare_dataset/get_npgraph.py" --data_dir="$data_dir" --graph_name="$GRAPH"
else
  echo "-- topology/idx/feat present --"
fi

# freq weights — only needed to (re)build partition maps
need_maps=0
for k in $WORLD_SIZES; do
  [ -f "$MDIR/${GRAPH}_w${k}_${PTYPE}.npy" ] || need_maps=1
done
if [ "$need_maps" = 1 ] && { [ ! -f "$GDIR/node_weight.npy" ] || [ ! -f "$GDIR/edge_weight.npy" ]; }; then
  echo "-- freq weights missing: get_weight_fast (fanout 15,15,15, ws=$WEIGHT_WS) --"
  # NOTE: get_weight_fast.py calls os._exit(1) even on success; gate on the output files.
  python3 "$PD/prepare_dataset/get_weight_fast.py" --data_dir="$data_dir" --graph_name="$GRAPH" \
    --fanouts=15,15,15 --num_epoch=5 --world_size="$WEIGHT_WS" || true
fi

# ---- partition maps ($PTYPE) for each requested world size ----
for k in $WORLD_SIZES; do
  MAP="$MDIR/${GRAPH}_w${k}_${PTYPE}.npy"
  if [ -f "$MAP" ]; then echo "map w$k present"; continue; fi
  echo "-- building partition map w$k (METIS, scheme=$PTYPE) --"
  # METIS can segfault nondeterministically on huge graphs; retry once.
  for attempt in 1 2; do
    python3 "$PD/prepare_dataset/get_partition.py" --data_dir="$data_dir" --graph_name="$GRAPH" \
      --num_partition="$k" --node_weight="$PART_NODE" --edge_weight="$PART_EDGE" --bal="$PART_BAL" \
      || echo "WARN get_partition k=$k attempt $attempt crashed"
    [ -f "$MAP" ] && break
    echo "   retrying map w$k ..."
  done
  [ -f "$MAP" ] || echo "WARN partition map w$k NOT produced"
done

# ---- 3. push to B2 ----
# Tier A2: partition maps are small + non-reproducible -> pushed by DEFAULT.
# The large graph/<g>/ dir regenerates from OGB and can hit the B2 cap -> opt-in (PUSH_B2=1).
if { [ "$PUSH_MAPS" = 1 ] || [ "$PUSH_B2" = 1 ]; } && ! have_b2; then
  echo "WARN B2 push requested but no '$B2_REMOTE' rclone remote — set up B2 creds per"
  echo "     RUNBOOK §Prerequisites (or run sync_code.sh --with-creds). Skipping push."
else
  if [ "$PUSH_MAPS" = 1 ] && have_b2; then
    echo "-- pushing partition maps to B2 (Tier A2) --"
    rclone copy "$MDIR" "$B2_REMOTE:$B2_PREFIX/partition_map/$GRAPH" -P || echo "WARN map push failed"
  fi
  if [ "$PUSH_B2" = 1 ] && have_b2; then
    echo "-- PUSH_B2=1: also pushing graph/$GRAPH to B2 --"
    rclone copy "$GDIR" "$B2_REMOTE:$B2_PREFIX/graph/$GRAPH" -P --transfers=16 --b2-chunk-size=100M || echo "WARN graph push failed (B2 storage cap?)"
  fi
fi

echo "== provision $GRAPH done =="
echo "graph dir:"; ls -la "$GDIR"
echo "maps:";      ls -la "$MDIR"
