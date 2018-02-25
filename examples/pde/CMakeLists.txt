cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-PDE
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_CPU_EXAMPLES)
  # Shallow Water simulation example
  add_executable(swe_cpu swe.cpp)
  target_link_libraries(swe_cpu ArrayFire::afcpu)
endif()

if(BUILD_CUDA_EXAMPLES)
  add_executable(swe_cuda swe.cpp)
  target_link_libraries(swe_cuda ArrayFire::afcuda)
endif()

if(BUILD_OPENCL_EXAMPLES)
  add_executable(swe_opencl swe.cpp)
  target_link_libraries(swe_opencl ArrayFire::afopencl)
endif()
