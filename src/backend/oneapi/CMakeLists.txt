#Copyright(c) 2022, ArrayFire
#All rights reserved.
#
#This file is distributed under 3 - clause BSD license.
#The complete license agreement can be obtained at:
#http:  // arrayfire.com/licenses/BSD-3-Clause

if (AF_BUILD_ONEAPI)
    if (WIN32)
    find_package(IntelSYCL REQUIRED)
    find_package(oneDPL REQUIRED)
#include_directories(${SYCL_INCLUDE_DIR} / sycl)
    set(OPENCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR}/sycl)
    set(OPENCL_LIBRARY_DIR ${SYCL_LIBRARY_DIR})
  else()
    enable_language(SYCL)
  endif()
endif()

include(InternalUtils)
include(build_cl2hpp)
include(FileToString)

add_library(afoneapi
  Array.cpp
  Array.hpp
  Event.cpp
  Event.hpp
  GraphicsResourceManager.cpp
  GraphicsResourceManager.hpp
  Module.hpp
  Param.cpp
  Param.hpp
  all.cpp
  anisotropic_diffusion.cpp
  anisotropic_diffusion.hpp
  any.cpp
  approx1.cpp
  approx2.cpp
  approx.hpp
  arith.hpp
  assign.cpp
  assign.hpp
  backend.hpp
  bilateral.cpp
  bilateral.hpp
  binary.hpp
  blas.cpp
  blas.hpp
  canny.cpp
  canny.hpp
  cast.hpp
  cholesky.cpp
  cholesky.hpp
  compile_module.cpp
  complex.hpp
  convolve.cpp
  convolve.hpp
  convolve_separable.cpp
  copy.cpp
  copy.hpp
  count.cpp
  device_manager.cpp
  device_manager.hpp
  diagonal.cpp
  diagonal.hpp
  diff.cpp
  diff.hpp
  err_oneapi.hpp
  errorcodes.cpp
  errorcodes.hpp
  exampleFunction.cpp
  exampleFunction.hpp
  fast.cpp
  fast.hpp
  fft.cpp
  fft.hpp
  fftconvolve.cpp
  fftconvolve.hpp
  flood_fill.cpp
  flood_fill.hpp
  gradient.cpp
  gradient.hpp
  harris.cpp
  harris.hpp
  hist_graphics.cpp
  hist_graphics.hpp
  histogram.cpp
  histogram.hpp
  homography.cpp
  homography.hpp
  hsv_rgb.cpp
  hsv_rgb.hpp
  identity.cpp
  identity.hpp
  iir.cpp
  iir.hpp
  image.cpp
  image.hpp
  index.cpp
  index.hpp
  inverse.cpp
  inverse.hpp
  iota.cpp
  iota.hpp
  ireduce.cpp
  ireduce.hpp
  jit.cpp
  jit/BufferNode.hpp
  jit/ShiftNode.hpp
  jit/kernel_generators.hpp
  join.cpp
  join.hpp
  logic.hpp
  lookup.cpp
  lookup.hpp
  lu.cpp
  lu.hpp
  match_template.cpp
  match_template.hpp
  math.cpp
  math.hpp
  max.cpp
  mean.cpp
  mean.hpp
  meanshift.cpp
  meanshift.hpp
  medfilt.cpp
  medfilt.hpp
  memory.cpp
  memory.hpp
  min.cpp
  minmax_op.hpp
  moments.cpp
  moments.hpp
  morph.cpp
  morph.hpp
  nearest_neighbour.cpp
  nearest_neighbour.hpp
  orb.cpp
  orb.hpp
  platform.cpp
  platform.hpp
  plot.cpp
  plot.hpp
  print.hpp
  product.cpp
  qr.cpp
  qr.hpp
  random_engine.cpp
  random_engine.hpp
  range.cpp
  range.hpp
  reduce.hpp
  reduce_impl.hpp
  regions.cpp
  regions.hpp
  reorder.cpp
  reorder.hpp
  reshape.cpp
  resize.cpp
  resize.hpp
  rotate.cpp
  rotate.hpp
  scalar.hpp
  scan.cpp
  scan.hpp
  scan_by_key.cpp
  scan_by_key.hpp
  select.cpp
  select.hpp
  set.cpp
  set.hpp
  shift.cpp
  shift.hpp
  sift.cpp
  sift.hpp
  sobel.cpp
  sobel.hpp
  solve.cpp
  solve.hpp
  sort.cpp
  sort.hpp
  sort_by_key.cpp
  sort_by_key.hpp
  sort_index.cpp
  sort_index.hpp
  sparse.cpp
  sparse.hpp
  sparse_arith.cpp
  sparse_arith.hpp
  sparse_blas.cpp
  sparse_blas.hpp
  sum.cpp
  surface.cpp
  surface.hpp
  susan.cpp
  susan.hpp
  svd.cpp
  svd.hpp
  tile.cpp
  tile.hpp
  topk.cpp
  topk.hpp
  transform.cpp
  transform.hpp
  transpose.cpp
  transpose_inplace.cpp
  transpose.hpp
  triangle.cpp
  triangle.hpp
  types.hpp
  unwrap.cpp
  unwrap.hpp
  vector_field.cpp
  vector_field.hpp
  where.cpp
  where.hpp
  wrap.cpp
  wrap.hpp
  )

target_sources(afoneapi
  PRIVATE
    kernel/KParam.hpp
    kernel/accessors.hpp
    kernel/approx1.hpp
    kernel/approx2.hpp
    kernel/assign.hpp
    kernel/bilateral.hpp
    kernel/convolve_separable.cpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/fftconvolve_common.hpp
    kernel/fftconvolve_multiply.hpp
    kernel/fftconvolve_pack.hpp
    kernel/fftconvolve_pad.hpp
    kernel/fftconvolve_reorder.hpp
    kernel/histogram.hpp
    kernel/iir.hpp
    kernel/identity.hpp
    kernel/interp.hpp
    kernel/iota.hpp
    kernel/ireduce.hpp
    kernel/lu_split.hpp
    kernel/memcopy.hpp
    kernel/mean.hpp
    kernel/pad_array_borders.hpp
    kernel/random_engine.hpp
    kernel/random_engine_write.hpp
    kernel/random_engine_mersenne.hpp
    kernel/random_engine_philox.hpp
    kernel/random_engine_threefry.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/reduce_all.hpp
    kernel/reduce_by_key.hpp
    kernel/reduce_first.hpp
    kernel/reduce_dim.hpp
    kernel/reorder.hpp
    kernel/scan_first.hpp
    kernel/scan_dim.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sparse.hpp
    kernel/sparse_arith.hpp
    kernel/transpose.hpp
    kernel/transpose_inplace.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/where.hpp
    kernel/wrap.hpp
    kernel/wrap_dilated.hpp
)

function(set_sycl_language)
  foreach(target ${ARGV})
    if (WIN32)
      get_target_property(target_type ${target} TYPE)
      if (NOT (${target_type} STREQUAL "INTERFACE_LIBRARY"))
        get_target_property(TGT_SOURCES ${target} SOURCES)
        add_sycl_to_target(TARGET ${target} SOURCES ${TGT_SOURCES})
      endif()
    else()
      set_target_properties(${target}
        PROPERTIES
          LINKER_LANGUAGE SYCL)

      get_target_property(TGT_SOURCES ${target} SOURCES)
      if(NOT TGT_SOURCES)
        get_target_property(TGT_SOURCES ${target} INTERFACE_SOURCES)
      endif()

      foreach(FILE ${TGT_SOURCES})
        get_filename_component(FILE_EXTENSION ${FILE} EXT)
        if(FILE_EXTENSION STREQUAL ".cpp")
          set_source_files_properties(${FILE} PROPERTIES LANGUAGE SYCL)
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

set(kernel_src
  ${CMAKE_CURRENT_SOURCE_DIR}/../opencl/kernel/KParam.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../opencl/kernel/jit.cl
)

set( kernel_headers_dir "kernel_headers")

file_to_string(
  SOURCES ${kernel_src}
  VARNAME kernel_files
  EXTENSION "hpp"
  OUTPUT_DIR ${kernel_headers_dir}
  TARGETS cl_kernel_targets
  NAMESPACE "arrayfire oneapi opencl"
)

add_dependencies(afoneapi ${cl_kernel_targets})

add_library(ArrayFire::afoneapi ALIAS afoneapi)

arrayfire_set_default_cxx_flags(afoneapi)

include("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/CMakeLists.txt")

target_include_directories(afoneapi
  SYSTEM PRIVATE
    ${SYCL_INCLUDE_DIR}
)

target_include_directories(afoneapi
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

target_compile_options(afoneapi
  PRIVATE
  $<$<COMPILE_LANGUAGE:SYCL>:
    -fno-sycl-id-queries-fit-in-int
    -sycl-std=2020
    $<$<PLATFORM_ID:Linux>: -fno-sycl-rdc>
    >
)

target_compile_definitions(afoneapi
  PRIVATE
    AF_ONEAPI
    WITH_LINEAR_ALGEBRA
    CL_TARGET_OPENCL_VERSION=300
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_HPP_MINIMUM_OPENCL_VERSION=110
    CL_HPP_ENABLE_EXCEPTIONS
  )

cmake_host_system_information(RESULT NumberOfThreads
  QUERY NUMBER_OF_LOGICAL_CORES)

target_link_libraries(afoneapi
  PRIVATE
    c_api_interface
    cpp_api_interface
    oneapi_sort_by_key
    afcommon_interface
    OpenCL::OpenCL
    OpenCL::cl2hpp
    -fno-sycl-id-queries-fit-in-int
    $<$<PLATFORM_ID:Linux>:-fsycl-link-huge-device-code>
    $<$<PLATFORM_ID:Linux>:-fvisibility-inlines-hidden>
    $<$<PLATFORM_ID:Linux>:-fno-sycl-rdc>
    -fsycl-max-parallel-link-jobs=${NumberOfThreads}
    MKL::MKL_DPCPP
  )
  set_sycl_language(afcommon_interface
    oneapi_sort_by_key
    c_api_interface
    cpp_api_interface
    afoneapi)

#af_split_debug_info(afoneapi ${AF_INSTALL_LIB_DIR})

install(TARGETS afoneapi
  EXPORT ArrayFireoneAPITargets
  COMPONENT oneapi
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR}
)

source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*)
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/src/backend/build_version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
source_group("" FILES CMakeLists.txt)
