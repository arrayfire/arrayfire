# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
project(ArrayFire-Example-Getting-Started
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  # Convolve examples
  add_executable(convolve_cpu convolve.cpp)
  target_link_libraries(convolve_cpu ArrayFire::afcpu)

  # Integer examples
  add_executable(integer_cpu integer.cpp)
  target_link_libraries(integer_cpu ArrayFire::afcpu)

  # Rainfall examples
  add_executable(rainfall_cpu rainfall.cpp)
  target_link_libraries(rainfall_cpu ArrayFire::afcpu)

  # Vectorization examples
  add_executable(vectorize_cpu vectorize.cpp)
  target_link_libraries(vectorize_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(convolve_cuda convolve.cpp)
  target_link_libraries(convolve_cuda ArrayFire::afcuda)

  add_executable(integer_cuda integer.cpp)
  target_link_libraries(integer_cuda ArrayFire::afcuda)

  add_executable(rainfall_cuda rainfall.cpp)
  target_link_libraries(rainfall_cuda ArrayFire::afcuda)

  add_executable(vectorize_cuda vectorize.cpp)
  target_link_libraries(vectorize_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(convolve_opencl convolve.cpp)
  target_link_libraries(convolve_opencl ArrayFire::afopencl)

  add_executable(integer_opencl integer.cpp)
  target_link_libraries(integer_opencl ArrayFire::afopencl)

  add_executable(rainfall_opencl rainfall.cpp)
  target_link_libraries(rainfall_opencl ArrayFire::afopencl)

  add_executable(vectorize_opencl vectorize.cpp)
  target_link_libraries(vectorize_opencl ArrayFire::afopencl)
endif()
