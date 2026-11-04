# Initial cache for the CUDA builds that the GPU tests run (cmake -C). The
# source tree can be any commit, including ones older than this repository's
# presets, so these settings come from here instead of the tree's own presets.
# They mirror the ci-linux-cuda preset, without examples and Forge.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_COMPILER gcc CACHE STRING "")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "")
set(AF_BUILD_CPU OFF CACHE BOOL "")
set(AF_BUILD_CUDA ON CACHE BOOL "")
set(AF_BUILD_OPENCL OFF CACHE BOOL "")
set(AF_BUILD_ONEAPI OFF CACHE BOOL "")
set(AF_BUILD_UNIFIED OFF CACHE BOOL "")
set(AF_BUILD_FORGE OFF CACHE BOOL "")
set(AF_BUILD_EXAMPLES OFF CACHE BOOL "")
set(AF_BUILD_DOCS OFF CACHE BOOL "")
set(AF_WITH_CUDNN OFF CACHE BOOL "")
# The Matrix Market tests, with the bundled data that ci.sh restores into the build tree.
include("${CMAKE_CURRENT_LIST_DIR}/matrixmarket.cmake")
set(AF_COMPUTE_LIBRARY "FFTW/LAPACK/BLAS" CACHE STRING "")
