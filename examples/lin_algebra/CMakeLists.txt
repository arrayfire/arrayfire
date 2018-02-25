cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_CPU_EXAMPLES)
  # Cholesky example
  add_executable(cholesky_cpu cholesky.cpp)
  target_link_libraries(cholesky_cpu ArrayFire::afcpu)

  # LU example
  add_executable(lu_cpu lu.cpp)
  target_link_libraries(lu_cpu ArrayFire::afcpu)

  # QR example
  add_executable(qr_cpu qr.cpp)
  target_link_libraries(qr_cpu ArrayFire::afcpu)

  # SVD example
  add_executable(svd_cpu svd.cpp)
  target_link_libraries(svd_cpu ArrayFire::afcpu)
endif()

if(BUILD_CUDA_EXAMPLES)
  add_executable(cholesky_cuda cholesky.cpp)
  target_link_libraries(cholesky_cuda ArrayFire::afcuda)

  add_executable(lu_cuda lu.cpp)
  target_link_libraries(lu_cuda ArrayFire::afcuda)

  add_executable(qr_cuda qr.cpp)
  target_link_libraries(qr_cuda ArrayFire::afcuda)

  add_executable(svd_cuda svd.cpp)
  target_link_libraries(svd_cuda ArrayFire::afcuda)
endif()

if(BUILD_OPENCL_EXAMPLES)
  add_executable(cholesky_opencl cholesky.cpp)
  target_link_libraries(cholesky_opencl ArrayFire::afopencl)

  add_executable(lu_opencl lu.cpp)
  target_link_libraries(lu_opencl ArrayFire::afopencl)

  add_executable(qr_opencl qr.cpp)
  target_link_libraries(qr_opencl ArrayFire::afopencl)

  add_executable(svd_opencl svd.cpp)
  target_link_libraries(svd_opencl ArrayFire::afopencl)
endif()
