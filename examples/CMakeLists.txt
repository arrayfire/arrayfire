# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Examples
  VERSION 3.5.0
  LANGUAGES CXX)

if(EXISTS "${CMAKE_MODULE_PATH}/ArrayFireExampleOverloads.cmake")
  include(ArrayFireExampleOverloads)
endif()

if (NOT ASSETS_DIR)
    set(ASSETS_DIR "" CACHE PATH "Data and images required for some examples (url: https://github.com/arrayfire/assets)")
endif (NOT ASSETS_DIR)

if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
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
