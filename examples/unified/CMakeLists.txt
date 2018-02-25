cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Example-Unified
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

#Include following file after find_package(arrayFire)
include(BuildOptions)

if(BUILD_UNIFIED_EXAMPLES)
  # Simple unified backend example
  add_executable(basic_unified basic.cpp)
  target_link_libraries(basic_unified ArrayFire::af)
endif()
