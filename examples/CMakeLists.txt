# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Examples
  VERSION 3.7.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
if(NOT EXISTS "${ArrayFire_SOURCE_DIR}/CMakeLists.txt")
  set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

file(TO_NATIVE_PATH ${ASSETS_DIR} ASSETS_DIR)

if(WIN32)
  string(REPLACE "\\" "\\\\" ASSETS_DIR  ${ASSETS_DIR})
  # - WIN32_LEAN_AND_MEAN & VC_EXTRALEAN reduces the number of
  #   windows headers being included.
  # - NOMINMAX is required for ArrayFire code that uses
  #   functions af::min & af::max. Having a namespace doesn't help also.
  add_definitions(-DWIN32_LEAN_AND_MEAN -DVC_EXTRALEAN -DNOMINMAX)
  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
endif()

add_subdirectory(benchmarks)
add_subdirectory(computer_vision)
add_subdirectory(financial)
add_subdirectory(getting_started)
add_subdirectory(graphics)
add_subdirectory(helloworld)
add_subdirectory(image_processing)
add_subdirectory(lin_algebra)
add_subdirectory(machine_learning)
add_subdirectory(pde)
add_subdirectory(unified)
