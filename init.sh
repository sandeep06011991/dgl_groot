#!/bin/bash
# Build the third-party native deps (GKlib, METIS, oneTBB, NCCL) into
# third_party/build/{include,lib}. build.sh runs this automatically when those
# artifacts are missing. Fail loudly instead of silently skipping a dep.
set -eo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# GKlib and METIS declare cmake_minimum_required(VERSION 2.8). CMake >= 4.0
# removed compatibility with minimums below 3.5, so their configure step dies
# with "Compatibility with CMake < 3.5 has been removed" unless we opt back in.
# Exporting this covers every cmake invocation below (GKlib, METIS, oneTBB).
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# rm -rf ${SCRIPT_DIR}/third_party/build

# build GKlib
pushd ${SCRIPT_DIR}/third_party/GKlib
# export GKLIB_PATH="${SCRIPT_DIR}/third_party/GKlib"
make CONFIG_FLAGS="-DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-D_POSIX_C_SOURCE=199309L -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -D CMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build" config
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
