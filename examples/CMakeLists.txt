cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.5)
project(ArrayFire-Examples
  VERSION 3.5.0
  LANGUAGES CXX)

if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

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

function(target_link_libraries target sources)
  _target_link_libraries(example_${target} ${sources})
endfunction()

function(target_compile_definitions target access definitions)
  _target_compile_definitions(example_${target} ${access} ${definitions})
endfunction()

function(find_package args)
  if(NOT (TARGET ArrayFire::afcpu OR TARGET ArrayFire::afcuda OR TARGET ArrayFire::afopencl OR TARGET ArrayFire::af))
    _find_package(args)
  endif()
endfunction()

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
