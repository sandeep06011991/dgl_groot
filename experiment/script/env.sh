#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

WORKSPACE_DIR="/scratch4/workspace/spolisetty_umass_edu-groot"


export dataset_dir=$(realpath ${WORKSPACE_DIR}/dataset/)
export data_dir=$(realpath ${WORKSPACE_DIR}/graph/)
export python_dir=$(realpath $SCRIPT_DIR/..)

echo "Dataset directory: $dataset_dir"
echo "Data directory: $data_dir"
echo "Python directory: $python_dir"
