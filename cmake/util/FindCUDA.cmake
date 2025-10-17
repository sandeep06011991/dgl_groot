#######################################################
# Enhanced version of find CUDA.
#
# Usage:
#   find_cuda(${USE_CUDA})
#
# - When USE_CUDA=ON, use auto search
#
# Please use the CMAKE variable CUDA_TOOLKIT_ROOT_DIR to set CUDA directory
#
# Provide variables:
#
# - CUDA_FOUND
# - CUDA_INCLUDE_DIRS
# - CUDA_TOOLKIT_ROOT_DIR
# - CUDA_CUDA_LIBRARY
# - CUDA_CUDART_LIBRARY
# - CUDA_NVRTC_LIBRARY
# - CUDA_CUDNN_LIBRARY
# - CUDA_CUBLAS_LIBRARY
#
macro(find_cuda use_cuda)
  set(__use_cuda ${use_cuda})
  if(__use_cuda STREQUAL "ON")
    include(FindCUDA)
  endif()

  # additional libraries
  if(CUDA_FOUND)
    if(MSVC)
      find_library(CUDA_CUDA_LIBRARY cuda
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
      find_library(CUDA_NVRTC_LIBRARY nvrtc
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
      find_library(CUDA_CUDNN_LIBRARY cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
      find_library(CUDA_CUBLAS_LIBRARY cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
      find_library(CUDA_CURAND_LIBRARY curand
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
    else(MSVC)
      # find_library(CUDA_CUDA_LIBRARY cuda
      #   ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
      # find_library(CUDA_CUBLAS_LIBRARY cublas
      #   ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      #   ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      # find_library(CUDA_CURAND_LIBRARY curand
      #   ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      #   ${CUDA_TOOLKIT_ROOT_DIR}/lib)

      set(CUDA_CUDA_LIBRARY $ENV{CUDA_HOME}/lib/stubs/libcuda.so)
      set(CUDA_CUBLAS_LIBRARY $ENV{CUDA_HOME}/lib/libcublas.so)
      set(CUDA_CURAND_LIBRARY $ENV{CUDA_HOME}/lib/libcurand.so)
      set(CUDA_cusparse_LIBRARY $ENV{CUDA_HOME}/lib/libcusparse.so)
      set(CUDA_CUDART_LIBRARY $ENV{CUDA_HOME}/lib/libcudart.so)
    endif(MSVC)
    message(STATUS "Found CUDA_TOOLKIT_ROOT_DIR=" ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "Found CUDA_CUDA_LIBRARY=" ${CUDA_CUDA_LIBRARY})
    message(STATUS "Found CUDA_CUDART_LIBRARY=" ${CUDA_CUDART_LIBRARY})
    message(STATUS "Found CUDA_NVRTC_LIBRARY=" ${CUDA_NVRTC_LIBRARY})
    message(STATUS "Found CUDA_CUDNN_LIBRARY=" ${CUDA_CUDNN_LIBRARY})
    message(STATUS "Found CUDA_CUBLAS_LIBRARY=" ${CUDA_CUBLAS_LIBRARY})
    message(STATUS "Found CUDA_CURAND_LIBRARY=" ${CUDA_CURAND_LIBRARY})
  endif(CUDA_FOUND)
endmacro(find_cuda)
