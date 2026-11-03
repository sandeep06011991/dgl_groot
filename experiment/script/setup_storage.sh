#!/usr/bin/env bash
# setup_storage.sh — RUN ON THE NODE, ONCE, at bring-up (before provision/build).
#
# Decides WHERE run data + result logs live, based on whether a persistent volume is
# actually mounted, and makes sure the data is present locally either way:
#
#   * persistent volume mounted  -> data + logs live ON the volume; they SURVIVE a
#     terminate, so a redeploy reuses them (only what's missing is pulled from B2).
#   * no persistent volume        -> data + logs live on the LOCAL (ephemeral) disk;
#     the script PULLS everything needed from Backblaze B2 now, and results MUST be
#     pushed back to B2 (or scp'd off) before terminate or they are lost.
#
# IMPORTANT (geesefs): MC's persistent volume is object-storage-backed (FUSE). You can
# keep DATA and RESULT files on it, but you CANNOT run a conda env or load compiled
# .so files off it. So the conda env + repo + build ALWAYS stay on local disk
# (rebuilt by cloud/runpod/bootstrap_pod.sh, or restored from a tarball) — only data+logs are routed
# to the volume. Do not move ENV_PATH/REPO_DIR onto the volume.
#
#   Usage:  setup_storage.sh [graph ...]        # e.g. setup_storage.sh products papers100M
#
# Env:
#   PERSIST_DIR  explicit persistent mount to use (skips autodetect)
#   GROOT_STATE  where the sourceable decision file is written (default /root/groot_storage.env)
#   WORLD_SIZES  world sizes whose maps to ensure when pulling data (default "4")
#   B2_REMOTE/B2_PREFIX  rclone remote + bucket prefix (defaults match provision_data.sh)
#   SKIP_PULL    1 = only decide + export paths, do not pull data from B2 (default 0)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHS=("$@")
: "${GROOT_STATE:=/root/groot_storage.env}"
: "${WORLD_SIZES:=4}"
: "${B2_REMOTE:=gsplit-b2}"
: "${B2_PREFIX:=gsplit-data/groot}"
: "${SKIP_PULL:=0}"

# ---- 1. detect a persistent mount ------------------------------------------------
# MC mounts a volume at /persistent/<name>. Treat a path as persistent only if it is a
# real mountpoint (not just a dir on the ephemeral OS disk) and is writable.
is_mount(){ findmnt -rn --target "$1" >/dev/null 2>&1 && \
            [ "$(findmnt -rno TARGET --target "$1" 2>/dev/null)" = "$1" ]; }
detect_persist(){
  if [ -n "${PERSIST_DIR:-}" ]; then echo "$PERSIST_DIR"; return; fi
  shopt -s nullglob
  for d in /persistent/*/ /mnt/persistent/*/ /persistent /mnt/persistent; do
    d="${d%/}"
    [ -d "$d" ] || continue
    if is_mount "$d" && [ -w "$d" ]; then echo "$d"; return; fi
  done
  # last resort: a writable /persistent dir even if findmnt can't confirm the mount
  [ -d /persistent ] && [ -w /persistent ] && { echo /persistent; return; }
  echo ""   # none
}

PERSIST="$(detect_persist)"
if [ -n "$PERSIST" ]; then
  EPHEMERAL=0
  GROOT_DATA_ROOT="$PERSIST/groot"
  echo "== persistent storage DETECTED at: $PERSIST =="
else
  EPHEMERAL=1
  GROOT_DATA_ROOT="/home/groot_local"
  echo "== NO persistent storage — using LOCAL ephemeral disk ($GROOT_DATA_ROOT) =="
  echo "   NOTE: results here are LOST on terminate. They must be pushed to B2 (or"
  echo "         scp'd off) before the box is killed."
fi

data_dir="$GROOT_DATA_ROOT/graph"          # provision_data.sh / train_main.py --data_dir
LOGS_DIR="$GROOT_DATA_ROOT/logs"           # result CSVs
mkdir -p "$data_dir" "$LOGS_DIR"

# conda env + repo ALWAYS local (geesefs can't exec) — keep bootstrap defaults
ENV_PATH="${ENV_PATH:-/home/groot_env}"
REPO_DIR="${REPO_DIR:-/home/dgl_groot}"

# ---- 2. write the sourceable decision file --------------------------------------
cat > "$GROOT_STATE" <<EOF
# groot storage decision — written by setup_storage.sh @ $(date -u +%FT%TZ)
export GROOT_EPHEMERAL=$EPHEMERAL
export GROOT_DATA_ROOT="$GROOT_DATA_ROOT"
export data_dir="$data_dir"
export GROOT_LOGS_DIR="$LOGS_DIR"
export ENV_PATH="$ENV_PATH"
export REPO_DIR="$REPO_DIR"
export B2_REMOTE="$B2_REMOTE"
export B2_PREFIX="$B2_PREFIX"
EOF
# make every future SSH shell pick it up
grep -q "source $GROOT_STATE" /root/.bashrc 2>/dev/null || echo "source $GROOT_STATE" >> /root/.bashrc
echo "-- wrote $GROOT_STATE (sourced from /root/.bashrc) --"
cat "$GROOT_STATE"

# ---- 3. ensure data is present locally (pull from B2) ---------------------------
# Persistent + already populated -> provision_data.sh no-ops. Ephemeral or empty
# persistent -> it restores topology+idx+maps from B2 (features skipped; random_feat).
if [ "$SKIP_PULL" = 1 ]; then
  echo "-- SKIP_PULL=1: not pulling data --"
elif [ ${#GRAPHS[@]} -eq 0 ]; then
  echo "-- no graphs given; skipping data pull (pass graphs to pull, e.g. 'products papers100M') --"
else
  for g in "${GRAPHS[@]}"; do
    echo "-- ensuring data for '$g' under $data_dir (B2-restore first) --"
    data_dir="$data_dir" B2_REMOTE="$B2_REMOTE" B2_PREFIX="$B2_PREFIX" \
      bash "$HERE/provision_data.sh" "$g" "$WORLD_SIZES" || {
        echo "ERROR: could not provision '$g' (see provision_data.sh output above)."
        echo "       If its data is not on B2, push it there once, or re-run with ALLOW_OGB=1/ALLOW_METIS=1."
        exit 3; }
  done
fi

echo
echo "== storage ready =="
echo "   ephemeral : $EPHEMERAL  (1 = results must be B2-pushed before terminate)"
echo "   data_dir  : $data_dir"
echo "   logs      : $LOGS_DIR"
echo "   env/repo  : $ENV_PATH / $REPO_DIR  (local disk — rebuilt/tarball, never on the volume)"
if [ "$EPHEMERAL" = 1 ]; then
  echo "   REMINDER  : before terminate, push results:"
  echo "               rclone copy \"$LOGS_DIR\" \"$B2_REMOTE:$B2_PREFIX/logs\" -P"
fi
