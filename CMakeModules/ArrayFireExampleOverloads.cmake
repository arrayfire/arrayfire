# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

# Some examples take too long to execute. This list is used to exclude these
# examples from the tests
list(APPEND exclude_from_tests  black_scholes_options_cpu
                                monte_carlo_options_cpu
                                vectorize_cpu
                                )

# Overload add_executable and target_link_libraries so that we can use simple
# CMakeLists.txt files for the examples.
#
# These functions will overload the existing functions so that the target names
# have the word "examples_" prefixed to them so they don't conflict with the
# tests. This is an issue with the blas example where the test blas_cpu and the
# example blas_cpu have the same target name.
#
# Additionally, This will allow us to write the CMakeLists.txt files as
# standalone files so that they are easier to parse for new users.
function(add_executable target sources)
  _add_executable(example_${target} ${sources})
  set_target_properties(example_${target}
    PROPERTIES
      OUTPUT_NAME ${target}
      FOLDER "Examples"
  )

  if(NOT ${target} IN_LIST exclude_from_tests)
    #add_test(example_${target} ${target} 0 -)
  endif()
endfunction()

macro(find_package)
  _find_package(${ARGV})
  if(DEFINED AF_BUILD_CPU AND NOT AF_BUILD_CPU)
    set(ArrayFire_CPU_FOUND OFF)
  endif()
  if(DEFINED AF_BUILD_CUDA AND NOT AF_BUILD_CUDA)
    set(ArrayFire_CUDA_FOUND OFF)
  endif()
  if(DEFINED AF_BUILD_OPENCL AND NOT AF_BUILD_OPENCL)
    set(ArrayFire_OpenCL_FOUND OFF)
  endif()
endmacro()

function(target_link_libraries target sources)
  _target_link_libraries(example_${target} ${sources})
endfunction()

function(target_compile_definitions target access definitions)
  _target_compile_definitions(example_${target} ${access} ${definitions})
endfunction()
