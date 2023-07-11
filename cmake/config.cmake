#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disable CUDA
# - /path/to/cuda: use specific path to cuda toolkit

set(USE_CUDA ON)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
set(CUDA_CUDART_LIBRARY /home/ubuntu/miniconda3/envs/groot/lib/)
set(CUDA_CUBLAS_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib)
set(CUDA_CURAND_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib)
# set(CUDA_cusparse_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcusparse.so)

#---------------------------------------------
# Misc.
#---------------------------------------------
# Whether to build cpp unittest executables.
set(BUILD_CPP_TEST OFF)

# Whether to enable OpenMP.
set(USE_OPENMP ON)

# Whether to build PyTorch plugins.
set(BUILD_TORCH ON)

# Whether to build DGL sparse library.
set(BUILD_SPARSE ON)
