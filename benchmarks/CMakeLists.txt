# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

if (NOT TARGET benchmark)
  # benchmark targets cmake version 2.8 which throws warnings for policy CMP0048 on
  # newer cmakes. This sets the default global setting for that policy.
  set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

  set(BENCHMARK_ENABLE_TESTING OFF
    CACHE INTERNAL "Required to avoid testing the benchmark library")
  set(BENCHMARK_ENABLE_INSTALL OFF
    CACHE INTERNAL "Required to turn off installation in projects embedding benchmark")
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF
      CACHE INTERNAL "Required to turn off GTEST TESTS")
  set(BENCHMARK_ENABLE_LTO ON
      CACHE INTERNAL "Enable link time optimization for benchmark")

  add_subdirectory(gbench EXCLUDE_FROM_ALL)
  set_target_properties(benchmark
    PROPERTIES
      FOLDER "ExternalProjectTargets/gbench")

  #hide benchmark library variables
  mark_as_advanced(
    BENCHMARK_CXX_FLAGS_COVERAGE
    BENCHMARK_EXE_LINKER_FLAGS_COVERAGE
    BENCHMARK_SHARED_LINKER_FLAGS_COVERAGE
    BENCHMARK_BUILD_32_BITS
    BENCHMARK_ENABLE_EXCEPTIONS
    BENCHMARK_USE_LIBCXX
    LIBRT
    )

endif ()

if(AF_BUILD_CPU)
  list(APPEND enabled_backends "cpu")
endif(AF_BUILD_CPU)

if(AF_BUILD_CUDA)
  list(APPEND enabled_backends "cuda")
endif(AF_BUILD_CUDA)

if(AF_BUILD_OPENCL)
  list(APPEND enabled_backends "opencl")
endif(AF_BUILD_OPENCL)

# Creates benchmarks for all backends
#
# Creates a benchmark program for each backend. Most of the time you only need to
# specify the name of the source file to create a test.
#
# Parameters
# ----------
# 'CXX11'       If set the tests will be compiled using c++11. Tests should strive
#               to be C++98 compilient
# 'SRC'         The source files for the test
# 'LIBRARIES'   Libraries other than ArrayFire that need to be linked
# 'DEFINITIONS' Definitions that need to be defined
# 'BACKENDS'    Backends to target for this test. If not set then the test will
#               compiled againat all backends
function(make_benchmark)
  set(options CXX11)
  set(single_args SRC)
  set(multi_args LIBRARIES DEFINITIONS BACKENDS)
  cmake_parse_arguments(mt_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  get_filename_component(src_name ${mt_args_SRC} NAME_WE)
  foreach(backend ${enabled_backends})
    if(NOT "${mt_args_BACKENDS}" STREQUAL "" AND
       NOT ${backend} IN_LIST mt_args_BACKENDS)
      continue()
    endif()
    set(target "benchmark_${src_name}_${backend}")
    add_executable(${target} ${mt_args_SRC})
    target_include_directories(${target}
      PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/gbench/include
        ${CMAKE_CURRENT_SOURCE_DIR}
      )

    target_link_libraries(${target}
      PRIVATE
        benchmark
        af${backend}
        ${mt_args_LIBRARIES}
      )

    if(${mt_args_CXX11})
      set_target_properties(${target}
        PROPERTIES CXX_STANDARD 11)
    endif(${mt_args_CXX11})

	set_target_properties(${target}
      PROPERTIES FOLDER "Benchmarks")

    target_compile_definitions(${target}
      PRIVATE
        AF_$<UPPER_CASE:${backend}>
        ${mt_args_DEFINITIONS}
      )

    if(WIN32)
      target_compile_definitions(${target}
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
  endforeach()
endfunction(make_benchmark)

make_benchmark(SRC arith/pi.cpp CXX11)
make_benchmark(SRC linalg/blas.cpp CXX11)
make_benchmark(SRC linalg/cg.cpp CXX11)
make_benchmark(SRC signal/fft.cpp CXX11)
