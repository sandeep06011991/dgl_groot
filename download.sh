#!/bin/bash
# These artifacts are now stale. S3 links are probably deleted.
# Skip for now.  
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/experiment/script/env.sh"
mkdir -p $dataset_dir
pushd $dataset_dir

wget https://spara-artifact-sc24.s3.us-east-2.amazonaws.com/dataset.tar.gz
tar -xvf dataset.tar.gz && mv numpy graph

wget https://spara-artifact-sc24.s3.us-east-2.amazonaws.com/partition_map.tar.gz
tar -xvf partition_map.tar.gz && mv numpy ./graph/partition_map

rm *.tar.gz
popd