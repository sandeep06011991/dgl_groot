#!/usr/bin/env bash
# sync_code.sh  —  RUN ON THE CONTROL MACHINE (laptop/WSL), not on a pod.
#
# Fast dev loop WITHOUT a git push/pull: rsync the local working tree straight to a
# live pod. Cold-start a pod with bootstrap_pod.sh (which git-clones once); after
# that, iterate with this. Python-only edits need no rebuild (just rerun); for
# C++/CUDA edits add --build to rebuild on the pod.
#
#   Usage:  sync_code.sh <ip:port> [--build] [--with-creds] [--dry-run]
#     or    POD_SSH=<ip:port> sync_code.sh [flags]
#   e.g.    sync_code.sh 1.2.3.4:40022 --build
#
# Flags:
#   --build        after sync, run ./build.sh release on the pod (native changes)
#   --with-creds   scp ~/.config/groot/b2.env to the pod and create the gsplit-b2
#                  rclone remote, so the pod can push maps/logs to B2 (see RUNBOOK
#                  §Prerequisites). The key lands only on the throwaway pod.
#   --dry-run      show what rsync would transfer, change nothing
#
# Env (defaults in parens):
#   POD_SSH     (none)        ip:port of the pod (positional arg or this env var)
#   SSH_KEY     (~/.runpod/ssh/runpodctl-ssh-key)   private key from `runpodctl ssh add-key`
#   REMOTE_DIR  (/home/dgl_groot)                   repo path on the pod
#   REMOTE_USER (root)
#   B2_ENV      (~/.config/groot/b2.env)            local B2 creds for --with-creds
#   DELETE      (0)           1 = rsync --delete (mirror; also removes pod-only files)
set -uo pipefail

BUILD=0; WITH_CREDS=0; DRY=0; POD_SSH="${POD_SSH:-}"
for a in "$@"; do
  case "$a" in
    --build) BUILD=1 ;;
    --with-creds) WITH_CREDS=1 ;;
    --dry-run) DRY=1 ;;
    --*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) POD_SSH="$a" ;;
  esac
done

: "${SSH_KEY:=$HOME/.runpod/ssh/runpodctl-ssh-key}"
: "${REMOTE_DIR:=/home/dgl_groot}"
: "${REMOTE_USER:=root}"
: "${B2_ENV:=$HOME/.config/groot/b2.env}"
: "${DELETE:=0}"

[ -n "$POD_SSH" ] || { echo "ERROR: POD_SSH not set. Pass <ip:port> or export POD_SSH (acquire_pod.sh prints it)." >&2; exit 2; }
[ -f "$SSH_KEY" ] || { echo "ERROR: SSH key '$SSH_KEY' not found. Run 'runpodctl ssh add-key' (RUNBOOK §Prerequisites)." >&2; exit 2; }

IP="${POD_SSH%%:*}"; PORT="${POD_SSH##*:}"
[ "$IP" != "$PORT" ] && [ -n "$PORT" ] || { echo "ERROR: POD_SSH must be ip:port (got '$POD_SSH')." >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_OPTS="-i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no"

RSYNC_ARGS=(-az --info=progress2 -e "ssh $SSH_OPTS"
  --exclude ".git/" --exclude "build/" --exclude "build-release/"
  --exclude "third_party/build/" --exclude "__pycache__/" --exclude "*.pyc"
  --exclude "*.so" --exclude "*.pth" --exclude "experiment/logs/"
  --exclude "dataset/" --exclude "datasets/" --exclude "graph/")
[ "$DELETE" = 1 ] && RSYNC_ARGS+=(--delete)
[ "$DRY" = 1 ] && RSYNC_ARGS+=(--dry-run)

echo "== rsync $REPO/ -> $REMOTE_USER@$IP:$REMOTE_DIR/  (delete=$DELETE dry=$DRY) =="
rsync "${RSYNC_ARGS[@]}" "$REPO/" "$REMOTE_USER@$IP:$REMOTE_DIR/"

if [ "$DRY" = 1 ]; then echo "(dry-run: no build/creds steps)"; exit 0; fi

if [ "$WITH_CREDS" = 1 ]; then
  if [ -f "$B2_ENV" ]; then
    echo "== copying B2 creds + creating gsplit-b2 remote on the pod =="
    scp $SSH_OPTS "$B2_ENV" "$REMOTE_USER@$IP:/root/b2.env"
    # shellcheck disable=SC2029  (expand B2_* on the pod from the copied file)
    ssh $SSH_OPTS "$REMOTE_USER@$IP" 'set -e
      command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | bash
      set -a; . /root/b2.env; set +a
      rclone config create gsplit-b2 b2 account "$B2_KEY_ID" key "$B2_APP_KEY" >/dev/null
      echo "   gsplit-b2 remote ready:"; rclone listremotes | grep "^gsplit-b2:"'
  else
    echo "WARN --with-creds but '$B2_ENV' not found; skipping (see RUNBOOK §Prerequisites)."
  fi
fi

if [ "$BUILD" = 1 ]; then
  echo "== building on the pod (./build.sh release) =="
  ssh $SSH_OPTS "$REMOTE_USER@$IP" "source /home/miniconda3/etc/profile.d/conda.sh && \
    conda activate /home/groot_env && cd $REMOTE_DIR && ./build.sh release"
fi

echo "== sync done -> $REMOTE_USER@$IP:$REMOTE_DIR =="
