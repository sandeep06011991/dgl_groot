#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# WORKSPACE_DIR defaults to the Unity path but is overridable (e.g. on a RunPod
# pod: export WORKSPACE_DIR=/root). realpath -m so a not-yet-created dir is fine.
WORKSPACE_DIR="${WORKSPACE_DIR:-/scratch3/workspace/spolisetty_umass_edu-gsplit}"

# data_dir may also be set directly (provision_data.sh default is /root/graph).
export dataset_dir="${dataset_dir:-$(realpath -m ${WORKSPACE_DIR}/dataset/)}"
export data_dir="${data_dir:-$(realpath -m ${WORKSPACE_DIR}/graph/)}"
export python_dir=$(realpath -m $SCRIPT_DIR/..)

echo "Dataset directory: $dataset_dir"
echo "Data directory: $data_dir"
echo "Python directory: $python_dir"
