#!/usr/bin/env bash
# mc_volume.sh — RUN ON A MASSED COMPUTE VM. Helper for the MC *persistent storage
# volume* (the beta "Persistent Storage" feature), the durable store that survives
# instance termination and is re-attached at the next launch.
#
# WHAT WE CONFIRMED ABOUT MC VOLUMES (2026-11, via the VM API + the live pod):
#   * The launch field is `volumeIds` (camelCase, array, max 8). `volume_ids` -> 422.
#   * Each attached volume mounts at  /persistent/<volume-name>  on the VM.
#   * Attach happens at LAUNCH ONLY. Attaching to an already-running VM updates the
#     volume's metadata (GET /storage-volumes shows attachedTo), but does NOT hot-mount:
#     on the live box /persistent/<name> stays empty / is not a mountpoint until the
#     VM is relaunched with that id in volumeIds. (This is exactly why a just-attached
#     volume is invisible on a running box.)
#   * The volume survives `instance/terminate`; it is billed (storage only) until the
#     volume itself is DELETEd. There is NO Stop that pauses instance billing on MC.
#   * The volume is object-storage-backed (the image ships /usr/local/bin/geesefs); the
#     conda env + compiled .so are NOT relocatable and must not run off it. So we use
#     the volume as a *tarball store*: snapshot env -> volume, and on a fresh VM copy
#     the tarball OFF the volume to local /home and untar there. Data .npy files MAY be
#     read straight off the mount.
#
# Layout on the volume mirrors the B2 layout (groot/{env,graph,partition_map,logs}):
#   /persistent/<name>/groot/env/<KEY>.tar.zst    (env + repo/build snapshot)
#   /persistent/<name>/groot/graph/<graph>/...    (dataset npy)
#
# Usage (on the VM):
#   bash mc_volume.sh detect                 # show the mounted volume + free space
#   bash mc_volume.sh snapshot-env           # verify import, tar env+repo -> volume
#   bash mc_volume.sh restore-env            # copy tarball off volume -> /home, untar, verify
#   bash mc_volume.sh snapshot-data <graph>  # rsync a graph's npy from $DATA_DIR -> volume
#   bash mc_volume.sh restore-data <graph>   # rsync a graph's npy from volume -> $DATA_DIR
#   bash mc_volume.sh seed-from-b2           # pull the env tarball from B2 -> volume (fresh volume)
# Env: VOL_MOUNT (auto: first /persistent/*), KEY (mc_a100_sm80),
#      ENV_PATH (/home/groot_env), REPO_DIR (/home/dgl_groot), DATA_DIR (/home/data)
set -uo pipefail

ENV_PATH="${ENV_PATH:-/home/groot_env}"
REPO_DIR="${REPO_DIR:-/home/dgl_groot}"
DATA_DIR="${DATA_DIR:-/home/data}"
KEY="${KEY:-mc_a100_sm80}"
MINICONDA="${MINICONDA:-/home/miniconda3}"

# Auto-detect the volume mount: first directory under /persistent that is a mountpoint
# (fall back to first /persistent/* dir if mountpoint(1) is unavailable).
detect_vol() {
  local d
  for d in /persistent/*/; do
    [ -d "$d" ] || continue
    if mountpoint -q "${d%/}" 2>/dev/null || [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      echo "${d%/}"; return 0
    fi
  done
  # last resort: any /persistent/* even if empty
  for d in /persistent/*/; do [ -d "$d" ] && { echo "${d%/}"; return 0; }; done
  return 1
}
VOL="${VOL_MOUNT:-$(detect_vol || true)}"
GROOT_DIR="${VOL:+$VOL/groot}"
TARBALL_ON_VOL="${GROOT_DIR:+$GROOT_DIR/env/$KEY.tar.zst}"

need_vol() {
  [ -n "$VOL" ] || { echo "ERROR: no persistent volume found under /persistent/ (launch with VOLUME_IDS=<id>)"; exit 3; }
}

cmd="${1:-detect}"
case "$cmd" in
  detect)
    echo "=== /persistent contents ==="; ls -la /persistent 2>/dev/null
    if [ -n "$VOL" ]; then
      echo "VOL_MOUNT=$VOL"
      mountpoint -q "$VOL" 2>/dev/null && echo "  (is a mountpoint)" || echo "  (NOT a mountpoint — volume may not be attached-at-launch)"
      df -hT "$VOL" 2>/dev/null | tail -n +1
      echo "env tarball: $TARBALL_ON_VOL"; [ -f "$TARBALL_ON_VOL" ] && ls -lh "$TARBALL_ON_VOL" || echo "  (absent)"
    else
      echo "NO volume mounted. On MC the volume mounts at /persistent/<name> and ONLY at launch."
      echo "Relaunch the VM with VOLUME_IDS=<id> (mc_launch.sh) to get it mounted."
    fi
    ;;

  snapshot-env)
    need_vol
    source "$MINICONDA/etc/profile.d/conda.sh" && conda activate "$ENV_PATH"
    export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"
    echo "=== verify import before snapshotting ==="
    python -c "import torch,dgl,torchmetrics; print('ok dgl',dgl.__path__)" \
      || { echo "ERROR: env does not import — refusing to snapshot a broken build"; exit 1; }
    command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not on PATH (apt-get install -y zstd)"; exit 1; }
    mkdir -p "$GROOT_DIR/env"
    tmp="/home/$KEY.tar.zst"
    echo "=== tar env+repo -> $tmp ==="
    tar -C / --zstd -cf "$tmp" home/groot_env home/dgl_groot
    ls -lh "$tmp"
    echo "=== copy onto volume: $TARBALL_ON_VOL ==="
    cp "$tmp" "$TARBALL_ON_VOL" && sync
    echo "snapshot on volume: $TARBALL_ON_VOL"
    ;;

  restore-env)
    # Returns 0 only if the restored env imports. Safe to call opportunistically.
    [ -n "$VOL" ] || { echo "restore-env: no volume mounted"; exit 3; }
    [ -f "$TARBALL_ON_VOL" ] || { echo "restore-env: no tarball at $TARBALL_ON_VOL"; exit 4; }
    command -v zstd >/dev/null 2>&1 || { echo "restore-env: zstd not on PATH (apt-get install -y zstd)"; exit 1; }
    tmp="/home/$KEY.tar.zst"
    echo "=== copy tarball OFF volume -> $tmp (cannot run env off the geesefs mount) ==="
    cp "$TARBALL_ON_VOL" "$tmp"
    echo "=== untar to local /home ==="
    tar -C / --zstd -xf "$tmp"
    source "$MINICONDA/etc/profile.d/conda.sh" && conda activate "$ENV_PATH"
    export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"
    python -c "import torch,dgl,torchmetrics; print('restored dgl',dgl.__path__)" \
      || { echo "ERROR: restored env does not import"; exit 1; }
    echo "restore-env OK (skip the build)"
    ;;

  snapshot-data)
    need_vol; graph="${2:?usage: mc_volume.sh snapshot-data <graph>}"
    mkdir -p "$GROOT_DIR/graph"
    echo "=== rsync $DATA_DIR/$graph -> $GROOT_DIR/graph/ ==="
    rsync -a --info=progress2 "$DATA_DIR/$graph" "$GROOT_DIR/graph/"
    ;;

  restore-data)
    need_vol; graph="${2:?usage: mc_volume.sh restore-data <graph>}"
    mkdir -p "$DATA_DIR"
    echo "=== rsync $GROOT_DIR/graph/$graph -> $DATA_DIR/ ==="
    rsync -a --info=progress2 "$GROOT_DIR/graph/$graph" "$DATA_DIR/"
    ;;

  seed-from-b2)
    # Seed a FRESH volume with the env tarball straight from B2 (no local build needed).
    # GOTCHAS (validated 2026-11-03): (1) you CANNOT `rclone copyto B2 <geesefs mount>` —
    # geesefs rejects rclone's chtimes ("operation not permitted"); (2) the local staging
    # dir must be WRITABLE — on a fresh VM /home is root-owned until mc_bootstrap chowns it,
    # so stage under $HOME. Recipe: rclone B2 -> $HOME, then plain `cp` -> volume.
    need_vol
    command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not installed"; exit 1; }
    rclone listremotes 2>/dev/null | grep -q '^gsplit-b2:' || { echo "ERROR: gsplit-b2 remote not configured (from ~/.config/groot/b2.env)"; exit 1; }
    mkdir -p "$GROOT_DIR/env"
    tmp="$HOME/$KEY.tar.zst"
    echo "=== download B2 env tarball -> $tmp (writable local dir) ==="
    rclone copyto "gsplit-b2:gsplit-data/groot/env/$KEY.tar.zst" "$tmp" --multi-thread-streams=4
    echo "=== cp -> volume (plain cp; geesefs rejects rclone's modtime set) ==="
    cp "$tmp" "$TARBALL_ON_VOL" && sync
    ls -l "$TARBALL_ON_VOL"
    ;;

  *)
    echo "usage: mc_volume.sh {detect|snapshot-env|restore-env|snapshot-data <g>|restore-data <g>|seed-from-b2}"; exit 2;;
esac
