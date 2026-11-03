#!/usr/bin/env bash
# acquire_pod.sh  —  RUN ON THE CONTROL MACHINE (laptop/WSL), not on a pod.
#
# A100-SXM capacity on RunPod is frequently "Low" or momentarily empty, so a single
# `pod create` often fails with "no longer any instances available". This retries
# across several datacenters until a pod is allocated, then prints how to reach it.
#
# Prereqs: runpodctl installed and authenticated (see RUNBOOK §"Provisioning a pod").
#
#   Usage:  acquire_pod.sh [pod-name]
#
# Env (defaults in parens):
#   GPU_ID        ("NVIDIA A100-SXM4-80GB")   gpu type, from `runpodctl gpu list`
#   GPU_COUNT     (2)                          #GPUs (split measurement needs >=2)
#   DISK_GB       (200)                         container disk
#   DCS           ("US-MD-1,US-MO-1,US-KS-2,US-WA-1,EUR-IS-1")  DCs to try (first avail wins)
#   IMAGE         (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
#   MAX_ATTEMPTS  (30)   RETRY_SECONDS (60)   WAIT_TIMEOUT (5m)
#
# On success: prints POD_ID, POD_SSH (ip:port), and a ready-to-use ssh command; exit 0.
# On exhaustion: exit 1.
set -uo pipefail

NAME="${1:-groot-$(date +%s)}"
GPU_ID="${GPU_ID:-NVIDIA A100-SXM4-80GB}"
GPU_COUNT="${GPU_COUNT:-2}"
DISK_GB="${DISK_GB:-200}"
# '-' (not ':-') so an explicit empty DCS="" is honored (-> search ALL datacenters),
# while an unset DCS still falls back to the known-good default list.
DCS="${DCS-US-MD-1,US-MO-1,US-KS-2,US-WA-1,EUR-IS-1}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5m}"

command -v runpodctl >/dev/null 2>&1 || { echo "ERROR: runpodctl not installed. See RUNBOOK §Prerequisites."; exit 2; }
# Preflight: an API key must be configured on THIS host (control machine). runpodctl
# stores it in ~/.runpod/config.toml; RUNPOD_API_KEY in the env also works.
if [ ! -f "$HOME/.runpod/config.toml" ] && [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "ERROR: no RunPod API key. Run 'runpodctl config --apiKey <key>' or export RUNPOD_API_KEY."
  echo "       See RUNBOOK §Prerequisites — credentials on the control machine."
  exit 2
fi

echo "Acquiring ${GPU_COUNT}x '${GPU_ID}' across [$DCS] (<= $MAX_ATTEMPTS tries, ${RETRY_SECONDS}s apart)"
# Empty DCS => omit --data-center-ids so RunPod allocates from ANY datacenter with
# capacity (the hardcoded 5-DC list is just a default; set DCS="" to search all).
DC_ARG=(); [ -n "$DCS" ] && DC_ARG=(--data-center-ids "$DCS")
# CLOUD_TYPE is a comma list of pools tried in order each attempt; first success wins
# and exits, so at most one pod is ever created. Default SECURE; "SECURE,COMMUNITY"
# also fishes the (larger, cheaper, less-reliable) community pool when secure is dry.
IFS=',' read -r -a CLOUD_TYPES <<< "${CLOUD_TYPE:-SECURE}"

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  out=""; won_ct=""; last=""
  for ct in "${CLOUD_TYPES[@]}"; do
    resp=$(runpodctl pod create --name "$NAME" --cloud-type "$ct" \
      --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" \
      "${DC_ARG[@]}" --container-disk-in-gb "$DISK_GB" \
      --image "$IMAGE" --ports "22/tcp" --wait --wait-timeout "$WAIT_TIMEOUT" 2>&1)
    if echo "$resp" | grep -q '"id"'; then out="$resp"; won_ct="$ct"; break; fi
    last="$resp"
  done
  if [ -n "$out" ]; then
    echo "$out" | python3 -c '
import json,sys
s=sys.stdin.read(); d=json.loads(s[s.find("{"):])
ssh=d["ssh"]
print("POD_ID="+d["id"])
print("POD_SSH=%s:%s"%(ssh["ip"],ssh["port"]))
print("INFO: dc=%s ram=%sGB gpus=%s cost=$%s/hr"%(d.get("machine",{}).get("dataCenterId"),d.get("memoryInGb"),d.get("gpuCount"),d.get("costPerHr")))
print("SSH: ssh -i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no root@%s -p %s"%(ssh["ip"],ssh["port"]))'
    echo "ACQUIRED on attempt $i (cloud=$won_ct)"
    exit 0
  fi
  if echo "$last" | grep -qiE "no longer any instances|no instances available|not available"; then
    echo "[$(date -u +%H:%M:%S) try $i/$MAX_ATTEMPTS] no capacity (tried: ${CLOUD_TYPES[*]}); retry in ${RETRY_SECONDS}s"
  else
    echo "[$(date -u +%H:%M:%S) try $i] unexpected response: $(echo "$last" | tail -n1)"
  fi
  sleep "$RETRY_SECONDS"
done
echo "FAILED: no $GPU_ID capacity after $MAX_ATTEMPTS attempts. Try again later, widen DCS, or lower GPU_COUNT."
exit 1
