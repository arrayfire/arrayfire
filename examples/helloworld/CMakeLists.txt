# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-HelloWorld
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  # Hello World example
  add_executable(helloworld_cpu helloworld.cpp)
  target_link_libraries(helloworld_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(helloworld_cuda helloworld.cpp)
  target_link_libraries(helloworld_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(helloworld_opencl helloworld.cpp)
  target_link_libraries(helloworld_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(helloworld_oneapi helloworld.cpp)
  target_link_libraries(helloworld_oneapi ArrayFire::afoneapi)
endif()
