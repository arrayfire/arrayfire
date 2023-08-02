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

if(ArrayFire_oneAPI_FOUND)
  add_executable(swe_oneapi swe.cpp)
  target_link_libraries(swe_oneapi ArrayFire::afoneapi)
endif()

if(ArrayFire_CPU_FOUND)
  # Smothered Particle Hydrodynamics Example
  add_executable(smh_cpu smothered_hydrodynamics.cpp)
  target_link_libraries(smh_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(smh_cuda smothered_hydrodynamics.cpp)
  target_link_libraries(smh_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(smh_opencl smothered_hydrodynamics.cpp)
  target_link_libraries(smh_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(smh_oneapi smothered_hydrodynamics.cpp)
  target_link_libraries(smh_oneapi ArrayFire::afoneapi)
endif()