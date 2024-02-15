#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "Script directory: $SCRIPT_DIR"
# build GKlib
cd ${SCRIPT_DIR}/third_party/GKlib && make config prefix=${SCRIPT_DIR}/metis_build openmp=set && make -j && make install
# build METIS
cd ${SCRIPT_DIR}/third_party/METIS && make config prefix=${SCRIPT_DIR}/metis_build i64=1 && make -j && make install
