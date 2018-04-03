# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
project(ArrayFire-Example-PDE
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  # Shallow Water simulation example
  add_executable(swe_cpu swe.cpp)
  target_link_libraries(swe_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(swe_cuda swe.cpp)
  target_link_libraries(swe_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(swe_opencl swe.cpp)
  target_link_libraries(swe_opencl ArrayFire::afopencl)
endif()
