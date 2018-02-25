cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-Financial
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_CPU_EXAMPLES)
  # Black-Scholes Options
  add_executable(black_scholes_options_cpu black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cpu ArrayFire::afcpu)

  # Heston Model
  add_executable(heston_model_cpu heston_model.cpp)
  target_link_libraries(heston_model_cpu ArrayFire::afcpu)

  # Monte Carlo Options
  add_executable(monte_carlo_options_cpu monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cpu ArrayFire::afcpu)
endif()

if(BUILD_CUDA_EXAMPLES)
  add_executable(black_scholes_options_cuda black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cuda ArrayFire::afcuda)

  add_executable(heston_model_cuda heston_model.cpp)
  target_link_libraries(heston_model_cuda ArrayFire::afcuda)

  add_executable(monte_carlo_options_cuda monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cuda ArrayFire::afcuda)
endif()

if(BUILD_OPENCL_EXAMPLES)
  add_executable(monte_carlo_options_opencl monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_opencl ArrayFire::afopencl)

  add_executable(black_scholes_options_opencl black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_opencl ArrayFire::afopencl)

  add_executable(heston_model_opencl heston_model.cpp)
  target_link_libraries(heston_model_opencl ArrayFire::afopencl)
endif()
