# Initial cache for the OpenCL builds that the GPU tests run (cmake -C). The
# source tree can be any commit, including ones older than this repository's
# presets, so these settings come from here instead of the tree's own presets.
# They mirror the ci-linux-opencl preset, without examples and Forge.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_COMPILER gcc CACHE STRING "")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "")
set(AF_BUILD_CPU OFF CACHE BOOL "")
set(AF_BUILD_CUDA OFF CACHE BOOL "")
set(AF_BUILD_OPENCL ON CACHE BOOL "")
set(AF_BUILD_ONEAPI OFF CACHE BOOL "")
set(AF_BUILD_UNIFIED OFF CACHE BOOL "")
set(AF_BUILD_FORGE OFF CACHE BOOL "")
set(AF_BUILD_EXAMPLES OFF CACHE BOOL "")
set(AF_BUILD_DOCS OFF CACHE BOOL "")
# The Matrix Market tests, with the bundled data that ci.sh restores into the build tree.
include("${CMAKE_CURRENT_LIST_DIR}/matrixmarket.cmake")
set(AF_COMPUTE_LIBRARY "FFTW/LAPACK/BLAS" CACHE STRING "")
