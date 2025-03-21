# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-PDE
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire REQUIRED)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(ArrayFire_CPU_FOUND)
  # Shallow Water simulation example
  add_executable(swe_cpu swe.cpp)
  target_link_libraries(swe_cpu ArrayFire::afcpu)

  # Black Hole Raytracing example
  add_executable(bhrt_cpu bhrt.cpp)
  target_link_libraries(bhrt_cpu ArrayFire::afcpu)

  add_executable(boltzmann_cfd_cpu boltzmann_cfd.cpp)
  target_link_libraries(boltzmann_cfd_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(swe_cuda swe.cpp)
  target_link_libraries(swe_cuda ArrayFire::afcuda)

  add_executable(bhrt_cuda bhrt.cpp)
  target_link_libraries(bhrt_cuda ArrayFire::afcuda)

  add_executable(boltzmann_cfd_cuda boltzmann_cfd.cpp)
  target_link_libraries(boltzmann_cfd_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(swe_opencl swe.cpp)
  target_link_libraries(swe_opencl ArrayFire::afopencl)

  add_executable(bhrt_opencl bhrt.cpp)
  target_link_libraries(bhrt_opencl ArrayFire::afopencl)

  add_executable(boltzmann_cfd_opencl boltzmann_cfd.cpp)
  target_link_libraries(boltzmann_cfd_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(swe_oneapi swe.cpp)
  target_link_libraries(swe_oneapi ArrayFire::afoneapi)

  add_executable(boltzmann_cfd_oneapi boltzmann_cfd.cpp)
  target_link_libraries(boltzmann_cfd_oneapi ArrayFire::afoneapi)
endif()
