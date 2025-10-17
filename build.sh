#!/bin/bash
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"

export CUDA_HOME=/work/pi_huiguan_umass_edu/sandeep/conda_environment/dgl_groot/
echo "CUDA_HOME=$CUDA_HOME"

bash init.sh

rm -rf build

cmake -B build -GNinja -DCMAKE_BUILD_TYPE=debug -DBUILD_TYPE=debug

cmake --build build -j

cd python && pip install . && cd ../

export LD_LIBRARY_PATH=${CUR_DIR}/third_party/build/lib:${LD_LIBRARY_PATH}
export CPATH=${CUR_DIR}/third_party/build/include:${CPATH}

cd third_party/torch-quiver && python3 setup.py build_ext --inplace && cd ../../

cd third_party/dist_cache/torch-quiver && python3 setup.py build_ext --inplace && cd ../../
