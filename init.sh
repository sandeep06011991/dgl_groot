#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# rm -rf ${SCRIPT_DIR}/third_party/build

# build GKlib
pushd ${SCRIPT_DIR}/third_party/GKlib
# export GKLIB_PATH="${SCRIPT_DIR}/third_party/GKlib"
make CONFIG_FLAGS="-DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-D_POSIX_C_SOURCE=199309L -D CMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build" config
make -j 
make install
popd

# build METIS
pushd ${SCRIPT_DIR}/third_party/metis
make config prefix=${SCRIPT_DIR}/third_party/build i64=1 gklib_path=${SCRIPT_DIR}/third_party/build #gdb=1 make -j
make install
popd

# build oneTBB
mkdir -p ${SCRIPT_DIR}/third_party/oneTBB/build
pushd ${SCRIPT_DIR}/third_party/oneTBB/build 
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build .. 
cmake --build . -j 
cmake --install . 
popd 

# build NCCL
pushd ${SCRIPT_DIR}/third_party/nccl
rm -rf ${SCRIPT_DIR}/third_party/nccl/build
make -j src.build NVCC_GENCODE="-arch=native" BUILDDIR=${SCRIPT_DIR}/third_party/build
popd
