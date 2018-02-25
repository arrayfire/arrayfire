cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-HelloWorld
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_CPU_EXAMPLES)
  # Hello World example
  add_executable(helloworld_cpu helloworld.cpp)
  target_link_libraries(helloworld_cpu ArrayFire::afcpu)
endif()

if(BUILD_CUDA_EXAMPLES)
  add_executable(helloworld_cuda helloworld.cpp)
  target_link_libraries(helloworld_cuda ArrayFire::afcuda)
endif()

if(BUILD_OPENCL_EXAMPLES)
  add_executable(helloworld_opencl helloworld.cpp)
  target_link_libraries(helloworld_opencl ArrayFire::afopencl)
endif()
