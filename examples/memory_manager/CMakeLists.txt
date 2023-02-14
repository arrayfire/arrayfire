# Copyright (c) 2022, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
project(ArrayFire-Example-Benchmarks
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  add_executable(memory_manager_cpu memory_manager.cpp
                                    ArrayFireDefaultMemoryManager.cpp
                                    ArrayFireDefaultMemoryManager.hpp)

  set_target_properties(memory_manager_cpu PROPERTIES CXX_STANDARD 11)
  target_link_libraries(memory_manager_cpu ArrayFire::afcpu)
endif()


if(ArrayFire_CUDA_FOUND)
  find_package(CUDA REQUIRED)
  add_executable(memory_manager_cuda memory_manager.cpp
                                     ArrayFireDefaultMemoryManager.cpp
                                     ArrayFireDefaultMemoryManager.hpp)

  set_target_properties(memory_manager_cuda PROPERTIES CXX_STANDARD 11)
  target_link_libraries(memory_manager_cuda ArrayFire::afcuda)
endif()


if(ArrayFire_OpenCL_FOUND)
  find_package(OpenCL REQUIRED)
  add_executable(memory_manager_opencl memory_manager.cpp
                                       ArrayFireDefaultMemoryManager.cpp
                                       ArrayFireDefaultMemoryManager.hpp)

  set_target_properties(memory_manager_opencl PROPERTIES CXX_STANDARD 11)
  target_link_libraries(memory_manager_opencl ArrayFire::afopencl)
endif()
