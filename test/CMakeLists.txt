# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

if(NOT TARGET gtest)
  # gtest targets cmake version 2.6 which throws warnings for policy CMP0042 on
  # newer cmakes. This sets the default global setting for that policy.
  set(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
  if(WIN32)
    set(gtest_force_shared_crt ON
        CACHE INTERNAL "Required so that the libs Runtime is not set to MT DLL")
  endif()

  add_subdirectory(gtest/googletest EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main
    PROPERTIES
      FOLDER "ExternalProjectTargets/gtest")

  # Hide gtest project variables
  mark_as_advanced(
    BUILD_SHARED_LIBS
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols)
endif()

# Reset the CXX flags for tests
set(CMAKE_CXX_STANDARD 98)
set(TESTDATA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

if(${AF_USE_RELATIVE_TEST_DIR})
    # RELATIVE_TEST_DATA_DIR is a User-visible option with default value of test/data directory
    set(RELATIVE_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE STRING "Relative Test Data Directory")
    set(TESTDATA_SOURCE_DIR ${RELATIVE_TEST_DATA_DIR})
else(${AF_USE_RELATIVE_TEST_DIR})  # Not using relative test data directory
    set(TESTDATA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif(${AF_USE_RELATIVE_TEST_DIR})

if(AF_BUILD_CPU)
  list(APPEND enabled_backends "cpu")
endif(AF_BUILD_CPU)

if(AF_BUILD_CUDA)
  list(APPEND enabled_backends "cuda")
endif(AF_BUILD_CUDA)

if(AF_BUILD_OPENCL)
  list(APPEND enabled_backends "opencl")
endif(AF_BUILD_OPENCL)

if(AF_BUILD_UNIFIED)
  list(APPEND enabled_backends "unified")
endif(AF_BUILD_UNIFIED)

include(CMakeParseArguments)

# Creates tests for all backends
#
# Creates a standard test for all backends. Most of the time you only need to
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
function(make_test)
  set(options CXX11 SERIAL)
  set(single_args SRC)
  set(multi_args LIBRARIES DEFINITIONS BACKENDS)
  cmake_parse_arguments(mt_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  get_filename_component(src_name ${mt_args_SRC} NAME_WE)
  foreach(backend ${enabled_backends})
    if(NOT "${mt_args_BACKENDS}" STREQUAL "" AND
       NOT ${backend} IN_LIST mt_args_BACKENDS)
      continue()
    endif()
    set(target "test_${src_name}_${backend}")
    add_executable(${target} ${mt_args_SRC})
    target_include_directories(${target}
      PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
      )
    target_link_libraries(${target}
      PRIVATE
        gtest
        gtest_main
        ${mt_args_LIBRARIES}
      )

    if(${backend} STREQUAL "unified")
      target_link_libraries(${target}
        PRIVATE
          af)
    else()
      target_link_libraries(${target}
        PRIVATE
          af${backend}
          )
    endif()

    if(${mt_args_CXX11})
      set_target_properties(${target}
        PROPERTIES
          CXX_STANDARD 11)
    endif(${mt_args_CXX11})

    set_target_properties(${target}
      PROPERTIES
        FOLDER "Tests"
        OUTPUT_NAME "${src_name}_${backend}")

    target_compile_definitions(${target}
      PRIVATE
        TEST_DIR="${TESTDATA_SOURCE_DIR}"
        AF_$<UPPER_CASE:${backend}>
        ${mt_args_DEFINITIONS}
      )
    if(WIN32)
      target_compile_options(${target}
        PRIVATE
          /bigobj
          /EHsc)
      target_compile_definitions(${target}
        PRIVATE
          WIN32_LEAN_AND_MEAN
          NOMINMAX)
    endif()

    # TODO(umar): Create this executable separately
    if(NOT ${backend} STREQUAL "unified" OR ${target} STREQUAL "backend_unified")
      add_test(NAME ${target} COMMAND ${target})
	    if(${mt_args_SERIAL})
        set_tests_properties(${target}
          PROPERTIES
            RUN_SERIAL ON)
	    endif(${mt_args_SERIAL})
    endif()

  endforeach()
endfunction(make_test)

make_test(SRC anisotropic_diffusion.cpp)
make_test(SRC approx1.cpp)
make_test(SRC approx2.cpp)
make_test(SRC array.cpp)
make_test(SRC arrayio.cpp)
make_test(SRC assign.cpp)
make_test(SRC backend.cpp)
make_test(SRC basic.cpp)
make_test(SRC basic_c.c)
make_test(SRC bilateral.cpp)
make_test(SRC binary.cpp)
make_test(SRC blas.cpp)
make_test(SRC canny.cpp)
make_test(SRC cast.cpp)
make_test(SRC cholesky_dense.cpp)
make_test(SRC clamp.cpp)
make_test(SRC compare.cpp)
make_test(SRC complex.cpp)
make_test(SRC constant.cpp)
make_test(SRC convolve.cpp)
make_test(SRC corrcoef.cpp)
make_test(SRC covariance.cpp)
make_test(SRC diagonal.cpp)
make_test(SRC diff1.cpp)
make_test(SRC diff2.cpp)
make_test(SRC dog.cpp)
make_test(SRC dot.cpp)
make_test(SRC empty.cpp)
make_test(SRC fast.cpp)
make_test(SRC fft.cpp)
make_test(SRC fft_large.cpp)
make_test(SRC fft_real.cpp)
make_test(SRC fftconvolve.cpp)
make_test(SRC flat.cpp)
make_test(SRC flip.cpp)
make_test(SRC gaussiankernel.cpp)
make_test(SRC gen_assign.cpp)
make_test(SRC gen_index.cpp)
make_test(SRC getting_started.cpp)
make_test(SRC gfor.cpp)
make_test(SRC gradient.cpp)
make_test(SRC gray_rgb.cpp)
make_test(SRC hamming.cpp)
make_test(SRC harris.cpp)
make_test(SRC histogram.cpp)
make_test(SRC homography.cpp)
make_test(SRC hsv_rgb.cpp)
make_test(SRC iir.cpp)
make_test(SRC imageio.cpp)
make_test(SRC index.cpp)
make_test(SRC info.cpp)
make_test(SRC internal.cpp)
make_test(SRC inverse_deconv.cpp)
make_test(SRC inverse_dense.cpp)
make_test(SRC iota.cpp)
make_test(SRC ireduce.cpp)
make_test(SRC iterative_deconv.cpp)
make_test(SRC jit.cpp)
make_test(SRC join.cpp)
make_test(SRC lu_dense.cpp)
make_test(SRC main.cpp)
#make_test(manual_memory_test.cpp)
make_test(SRC match_template.cpp)
make_test(SRC math.cpp)
make_test(SRC matrix_manipulation.cpp)
make_test(SRC mean.cpp)
make_test(SRC meanshift.cpp)
make_test(SRC meanvar.cpp CXX11)
make_test(SRC medfilt.cpp)
make_test(SRC median.cpp)
make_test(SRC memory.cpp)
make_test(SRC memory_lock.cpp)
make_test(SRC missing.cpp)
make_test(SRC moddims.cpp)
make_test(SRC moments.cpp)
make_test(SRC morph.cpp)
make_test(SRC nearest_neighbour.cpp)

if(OpenCL_FOUND)
  make_test(SRC ocl_ext_context.cpp
            LIBRARIES OpenCL::OpenCL
            BACKENDS "opencl")
endif()

make_test(SRC orb.cpp)
make_test(SRC qr_dense.cpp)
make_test(SRC random.cpp)
make_test(SRC range.cpp)
make_test(SRC rank_dense.cpp)
make_test(SRC reduce.cpp)
make_test(SRC regions.cpp)
make_test(SRC reorder.cpp)
make_test(SRC replace.cpp)
make_test(SRC resize.cpp)
make_test(SRC rotate.cpp)
make_test(SRC rotate_linear.cpp)
make_test(SRC sat.cpp)
make_test(SRC scan.cpp)
make_test(SRC scan_by_key.cpp)
make_test(SRC select.cpp)
make_test(SRC set.cpp)
make_test(SRC shift.cpp)

if(AF_WITH_NONFREE)
  make_test(SRC gloh_nonfree.cpp DEFINITIONS AF_WITH_NONFREE_SIFT)
  make_test(SRC sift_nonfree.cpp DEFINITIONS AF_WITH_NONFREE_SIFT)
endif()

make_test(SRC sobel.cpp)
make_test(SRC solve_dense.cpp       CXX11)
make_test(SRC sort.cpp)
make_test(SRC sort_by_key.cpp)
make_test(SRC sort_index.cpp)
make_test(SRC sparse.cpp            SERIAL)
make_test(SRC sparse_arith.cpp)
make_test(SRC sparse_convert.cpp)
make_test(SRC stdev.cpp)
make_test(SRC susan.cpp)
make_test(SRC svd_dense.cpp)
make_test(SRC threading.cpp         CXX11 SERIAL)
make_test(SRC tile.cpp)
make_test(SRC topk.cpp              CXX11)
make_test(SRC transform.cpp)
make_test(SRC transform_coordinates.cpp)
make_test(SRC translate.cpp)
make_test(SRC transpose.cpp)
make_test(SRC transpose_inplace.cpp)
make_test(SRC triangle.cpp)
make_test(SRC unwrap.cpp)
make_test(SRC var.cpp)
make_test(SRC where.cpp)
make_test(SRC wrap.cpp)
make_test(SRC write.cpp)
make_test(SRC ycbcr_rgb.cpp)


add_executable(print_info print_info.cpp)
if(AF_BUILD_UNIFIED)
  target_link_libraries(print_info ArrayFire::af)
elseif(AF_BUILD_OPENCL)
  target_link_libraries(print_info ArrayFire::afopencl)
elseif(AF_BUILD_CUDA)
  target_link_libraries(print_info ArrayFire::afcuda)
elseif(AF_BUILD_CPU)
  target_link_libraries(print_info ArrayFire::afcpu)
endif()
