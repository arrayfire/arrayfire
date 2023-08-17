# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
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

if(ArrayFire_CUDA_FOUND)
  add_executable(cholesky_cuda cholesky.cpp)
  target_link_libraries(cholesky_cuda ArrayFire::afcuda)

  add_executable(lu_cuda lu.cpp)
  target_link_libraries(lu_cuda ArrayFire::afcuda)

  add_executable(qr_cuda qr.cpp)
  target_link_libraries(qr_cuda ArrayFire::afcuda)

  add_executable(svd_cuda svd.cpp)
  target_link_libraries(svd_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(cholesky_opencl cholesky.cpp)
  target_link_libraries(cholesky_opencl ArrayFire::afopencl)

  add_executable(lu_opencl lu.cpp)
  target_link_libraries(lu_opencl ArrayFire::afopencl)

  add_executable(qr_opencl qr.cpp)
  target_link_libraries(qr_opencl ArrayFire::afopencl)

  add_executable(svd_opencl svd.cpp)
  target_link_libraries(svd_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(cholesky_oneapi cholesky.cpp)
  target_link_libraries(cholesky_oneapi ArrayFire::afoneapi)

  add_executable(lu_oneapi lu.cpp)
  target_link_libraries(lu_oneapi ArrayFire::afoneapi)

  add_executable(qr_oneapi qr.cpp)
  target_link_libraries(qr_oneapi ArrayFire::afoneapi)

  add_executable(svd_oneapi svd.cpp)
  target_link_libraries(svd_oneapi ArrayFire::afoneapi)
endif()
