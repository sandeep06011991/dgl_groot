#!/usr/bin/env bash
# mc_terminate.sh — RUN ON THE CONTROL MACHINE. Terminate Massed Compute instance(s) to
# STOP BILLING. WARNING: terminate destroys the instance disk (a plain Stop keeps billing).
# Back up / scp / commit any results off the node first. Persistent-storage volumes (if any)
# survive termination.
#
# Prereqs: export MC_API_TOKEN=<token>
#   Usage:  mc_terminate.sh <instance-uuid> [<instance-uuid> ...]
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
[ -n "${MC_API_TOKEN:-}" ] || { echo "ERROR: export MC_API_TOKEN=<token> first"; exit 2; }
[ "$#" -ge 1 ] || { echo "usage: mc_terminate.sh <instance-uuid> [...]"; exit 2; }
ids=$(printf '"%s",' "$@" | sed 's/,$//')
echo ">>> terminating: $*"
curl -s --max-time 30 -X POST -H "Authorization: Bearer $MC_API_TOKEN" -H "Content-Type: application/json" \
  -d "{\"instanceUuids\":[$ids]}" "$API/instance/terminate" -w $'\nHTTP %{http_code}\n'
