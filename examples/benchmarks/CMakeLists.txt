cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-Benchmarks
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_CPU_EXAMPLES)
  add_executable(blas_cpu blas.cpp)
  target_link_libraries(blas_cpu ArrayFire::afcpu)

  add_executable(cg_cpu cg.cpp)
  target_link_libraries(cg_cpu ArrayFire::afcpu)

  add_executable(fft_cpu fft.cpp)
  target_link_libraries(fft_cpu ArrayFire::afcpu)

  add_executable(pi_cpu pi.cpp)
  target_link_libraries(pi_cpu ArrayFire::afcpu)
endif()


if(BUILD_CUDA_EXAMPLES)
  add_executable(blas_cuda blas.cpp)
  target_link_libraries(blas_cuda ArrayFire::afcuda)

  add_executable(cg_cuda cg.cpp)
  target_link_libraries(cg_cuda ArrayFire::afcuda)

  add_executable(fft_cuda fft.cpp)
  target_link_libraries(fft_cuda ArrayFire::afcuda)

  add_executable(pi_cuda pi.cpp)
  target_link_libraries(pi_cuda ArrayFire::afcuda)
endif()


if(BUILD_OPENCL_EXAMPLES)
  add_executable(blas_opencl blas.cpp)
  target_link_libraries(blas_opencl ArrayFire::afopencl)

  add_executable(cg_opencl cg.cpp)
  target_link_libraries(cg_opencl ArrayFire::afopencl)

  add_executable(fft_opencl fft.cpp)
  target_link_libraries(fft_opencl ArrayFire::afopencl)

  add_executable(pi_opencl pi.cpp)
  target_link_libraries(pi_opencl ArrayFire::afopencl)
endif()
