# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-Graphics
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(ArrayFire_CPU_FOUND)
  # Conway Game of Life
  add_executable(conway_cpu conway.cpp)
  target_link_libraries(conway_cpu ArrayFire::afcpu)

  # Conway Game of Life with Color
  add_executable(conway_pretty_cpu conway_pretty.cpp)
  target_link_libraries(conway_pretty_cpu ArrayFire::afcpu)

  # Vector fields example
  add_executable(field_cpu field.cpp)
  target_link_libraries(field_cpu ArrayFire::afcpu)

  # Fractal example
  add_executable(fractal_cpu fractal.cpp)
  target_link_libraries(fractal_cpu ArrayFire::afcpu)

  # Gravity Simulation example
  add_executable(gravity_sim_cpu gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cpu ArrayFire::afcpu)

  # Histogram example
  add_executable(histogram_cpu histogram.cpp)
  target_compile_definitions(histogram_cpu PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cpu ArrayFire::afcpu)

  # Plot 2D example
  add_executable(plot2d_cpu plot2d.cpp)
  target_link_libraries(plot2d_cpu ArrayFire::afcpu)

  # Plot 3 example
  add_executable(plot3_cpu plot3.cpp)
  target_link_libraries(plot3_cpu ArrayFire::afcpu)

  # Surface example
  add_executable(surface_cpu surface.cpp)
  target_link_libraries(surface_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(conway_cuda conway.cpp)
  target_link_libraries(conway_cuda ArrayFire::afcuda)

  add_executable(conway_pretty_cuda conway_pretty.cpp)
  target_link_libraries(conway_pretty_cuda ArrayFire::afcuda)

  add_executable(field_cuda field.cpp)
  target_link_libraries(field_cuda ArrayFire::afcuda)

  add_executable(fractal_cuda fractal.cpp)
  target_link_libraries(fractal_cuda ArrayFire::afcuda)

  add_executable(gravity_sim_cuda gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cuda ArrayFire::afcuda)

  add_executable(histogram_cuda histogram.cpp)
  target_compile_definitions(histogram_cuda PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cuda ArrayFire::afcuda)

  add_executable(plot2d_cuda plot2d.cpp)
  target_link_libraries(plot2d_cuda ArrayFire::afcuda)
  add_executable(plot3_cuda plot3.cpp)
  target_link_libraries(plot3_cuda ArrayFire::afcuda)

  add_executable(surface_cuda surface.cpp)
  target_link_libraries(surface_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(conway_opencl conway.cpp)
  target_link_libraries(conway_opencl ArrayFire::afopencl)

  add_executable(conway_pretty_opencl conway_pretty.cpp)
  target_link_libraries(conway_pretty_opencl ArrayFire::afopencl)

  add_executable(field_opencl field.cpp)
  target_link_libraries(field_opencl ArrayFire::afopencl)

  add_executable(fractal_opencl fractal.cpp)
  target_link_libraries(fractal_opencl ArrayFire::afopencl)

  add_executable(gravity_sim_opencl gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_opencl ArrayFire::afopencl)

  add_executable(histogram_opencl histogram.cpp)
  target_compile_definitions(histogram_opencl PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_opencl ArrayFire::afopencl)

  add_executable(plot2d_opencl plot2d.cpp)
  target_link_libraries(plot2d_opencl ArrayFire::afopencl)

  add_executable(plot3_opencl plot3.cpp)
  target_link_libraries(plot3_opencl ArrayFire::afopencl)

  add_executable(surface_opencl surface.cpp)
  target_link_libraries(surface_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(conway_oneapi conway.cpp)
  target_link_libraries(conway_oneapi ArrayFire::afoneapi)

  add_executable(conway_pretty_oneapi conway_pretty.cpp)
  target_link_libraries(conway_pretty_oneapi ArrayFire::afoneapi)

  add_executable(field_oneapi field.cpp)
  target_link_libraries(field_oneapi ArrayFire::afoneapi)

  add_executable(fractal_oneapi fractal.cpp)
  target_link_libraries(fractal_oneapi ArrayFire::afoneapi)

  add_executable(gravity_sim_oneapi gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_oneapi ArrayFire::afoneapi)

  add_executable(histogram_oneapi histogram.cpp)
  target_compile_definitions(histogram_oneapi PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_oneapi ArrayFire::afoneapi)

  add_executable(plot2d_oneapi plot2d.cpp)
  target_link_libraries(plot2d_oneapi ArrayFire::afoneapi)

  add_executable(plot3_oneapi plot3.cpp)
  target_link_libraries(plot3_oneapi ArrayFire::afoneapi)

  add_executable(surface_oneapi surface.cpp)
  target_link_libraries(surface_oneapi ArrayFire::afoneapi)
endif()
