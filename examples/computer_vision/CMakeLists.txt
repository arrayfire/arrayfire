# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-Computer-Vision
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if (ArrayFire_CPU_FOUND)
  # FAST examples
  add_executable(fast_cpu fast.cpp)
  target_link_libraries(fast_cpu ArrayFire::afcpu)

  # Harris corner detector examples
  add_executable(harris_cpu harris.cpp)
  target_link_libraries(harris_cpu ArrayFire::afcpu)

  # Template Matching examples
  add_executable(matching_cpu matching.cpp)
  target_link_libraries(matching_cpu ArrayFire::afcpu)

  # Template Matching examples
  add_executable(susan_cpu susan.cpp)
  target_link_libraries(susan_cpu ArrayFire::afcpu)
endif()

if (ArrayFire_CUDA_FOUND)
  add_executable(fast_cuda fast.cpp)
  target_link_libraries(fast_cuda ArrayFire::afcuda)

  add_executable(harris_cuda harris.cpp)
  target_link_libraries(harris_cuda ArrayFire::afcuda)

  add_executable(matching_cuda matching.cpp)
  target_link_libraries(matching_cuda ArrayFire::afcuda)

  add_executable(susan_cuda susan.cpp)
  target_link_libraries(susan_cuda ArrayFire::afcuda)
endif()

if (ArrayFire_OpenCL_FOUND)
  add_executable(fast_opencl fast.cpp)
  target_link_libraries(fast_opencl ArrayFire::afopencl)

  add_executable(harris_opencl harris.cpp)
  target_link_libraries(harris_opencl ArrayFire::afopencl)

  add_executable(matching_opencl matching.cpp)
  target_link_libraries(matching_opencl ArrayFire::afopencl)

  add_executable(susan_opencl susan.cpp)
  target_link_libraries(susan_opencl ArrayFire::afopencl)
endif()

if (ArrayFire_oneAPI_FOUND)
  add_executable(fast_oneapi fast.cpp)
  target_link_libraries(fast_oneapi ArrayFire::afoneapi)

  add_executable(harris_oneapi harris.cpp)
  target_link_libraries(harris_oneapi ArrayFire::afoneapi)

  add_executable(matching_oneapi matching.cpp)
  target_link_libraries(matching_oneapi ArrayFire::afoneapi)

  add_executable(susan_oneapi susan.cpp)
  target_link_libraries(susan_oneapi ArrayFire::afoneapi)
endif()
