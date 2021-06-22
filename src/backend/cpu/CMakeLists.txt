# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)

generate_product_version(af_cpu_ver_res_file
  FILE_NAME "afcpu"
  FILE_DESCRIPTION "CPU Backend Dynamic-link library"
)

add_library(afcpu "")
add_library(ArrayFire::afcpu ALIAS afcpu)

# CPU backend source files
target_sources(afcpu
  PRIVATE
    $<$<PLATFORM_ID:Windows>:${af_cpu_ver_res_file}>
    Array.cpp
    Array.hpp
    anisotropic_diffusion.cpp
    anisotropic_diffusion.hpp
    approx.cpp
    approx.hpp
    arith.hpp
    assign.cpp
    assign.hpp
    backend.hpp
    bilateral.cpp
    bilateral.hpp
    blas.cpp
    blas.hpp
    canny.cpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    complex.hpp
    convolve.cpp
    convolve.hpp
    copy.cpp
    copy.hpp
    device_manager.cpp
    device_manager.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    err_cpu.hpp
    Event.cpp
    Event.hpp
    exampleFunction.cpp
    exampleFunction.hpp
    fast.cpp
    fast.hpp
    fft.cpp
    fft.hpp
    fftconvolve.cpp
    fftconvolve.hpp
    flood_fill.hpp
    flood_fill.cpp
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
    join.cpp
    join.hpp
    lapack_helper.hpp
    logic.hpp
    lookup.cpp
    lookup.hpp
    lu.cpp
    lu.hpp
    match_template.cpp
    match_template.hpp
    math.cpp
    math.hpp
    mean.cpp
    mean.hpp
    meanshift.cpp
    meanshift.hpp
    medfilt.cpp
    medfilt.hpp
    memory.cpp
    memory.hpp
    moments.cpp
    moments.hpp
    morph.cpp
    morph.hpp
    nearest_neighbour.cpp
    nearest_neighbour.hpp
    orb.cpp
    orb.hpp
    ParamIterator.hpp
    platform.cpp
    platform.hpp
    plot.cpp
    plot.hpp
    print.hpp
    qr.cpp
    qr.hpp
    queue.hpp
    random_engine.cpp
    random_engine.hpp
    range.cpp
    range.hpp
    reduce.cpp
    reduce.hpp
    regions.cpp
    regions.hpp
    reorder.cpp
    reorder.hpp
    resize.cpp
    resize.hpp
    reshape.cpp
    rotate.cpp
    rotate.hpp
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
    traits.hpp
    transform.cpp
    transform.hpp
    transpose.cpp
    transpose.hpp
    triangle.cpp
    triangle.hpp
    types.hpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    utility.hpp
    vector_field.cpp
    vector_field.hpp
    where.cpp
    where.hpp
    wrap.cpp
    wrap.hpp
  )

# CPU backend kernel files
target_sources(afcpu
  PRIVATE
    kernel/Array.hpp
    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/convolve.hpp
    kernel/copy.hpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/dot.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fftconvolve.hpp
    kernel/flood_fill.hpp
    kernel/gradient.hpp
    kernel/harris.hpp
    kernel/histogram.hpp
    kernel/hsv_rgb.hpp
    kernel/identity.hpp
    kernel/iir.hpp
    kernel/index.hpp
    kernel/interp.hpp
    kernel/iota.hpp
    kernel/ireduce.hpp
    kernel/join.hpp
    kernel/lookup.hpp
    kernel/lu.hpp
    kernel/match_template.hpp
    kernel/meanshift.hpp
    kernel/medfilt.hpp
    kernel/moments.hpp
    kernel/morph.hpp
    kernel/nearest_neighbour.hpp
    kernel/orb.hpp
    kernel/pad_array_borders.hpp
    kernel/random_engine.hpp
    kernel/random_engine_mersenne.hpp
    kernel/random_engine_philox.hpp
    kernel/random_engine_threefry.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/regions.hpp
    kernel/reorder.hpp
    kernel/resize.hpp
    kernel/rotate.hpp
    kernel/scan.hpp
    kernel/scan_by_key.hpp
    kernel/select.hpp
    kernel/shift.hpp
    kernel/sift.hpp
    kernel/sobel.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sort_helper.hpp
    kernel/sparse.hpp
    kernel/sparse_arith.hpp
    kernel/susan.hpp
    kernel/tile.hpp
    kernel/transform.hpp
    kernel/transpose.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/wrap.hpp
  )

if (AF_WITH_CPUID)
  target_compile_definitions(afcpu PRIVATE -DAF_WITH_CPUID)
endif(AF_WITH_CPUID)

FetchContent_Declare(
  ${threads_prefix}
  GIT_REPOSITORY https://github.com/arrayfire/threads.git
  GIT_TAG        b666773940269179f19ef11c8f1eb77005e85d9a
)
af_dep_check_and_populate(${threads_prefix})

target_sources(afcpu
  PRIVATE
    ${${threads_prefix}_SOURCE_DIR}/include/threads/async_queue.hpp
    ${${threads_prefix}_SOURCE_DIR}/include/threads/event.hpp
  )

arrayfire_set_default_cxx_flags(afcpu)

include("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/CMakeLists.txt")

target_include_directories(afcpu
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${${threads_prefix}_SOURCE_DIR}/include
    ${CBLAS_INCLUDE_DIR}
  )

target_compile_definitions(afcpu
  PRIVATE
    AF_CPU
  )

target_link_libraries(afcpu
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    cpu_sort_by_key
    Threads::Threads
  )
if(BUILD_WITH_MKL)
  target_compile_definitions(afcpu PRIVATE USE_MKL)
  if(AF_WITH_STATIC_MKL)
      target_link_libraries(afcpu PRIVATE MKL::Static)
  else()
      target_link_libraries(afcpu PRIVATE MKL::RT)
  endif()
else()
  target_link_libraries(afcpu
    PRIVATE
      ${CBLAS_LIBRARIES}
      FFTW::FFTW
      FFTW::FFTWF
    )
  if(LAPACK_FOUND)
    target_link_libraries(afcpu PRIVATE ${LAPACK_LIBRARIES})
    target_include_directories(afcpu PRIVATE ${LAPACK_INCLUDE_DIR})
  endif()
endif()

if(LAPACK_FOUND OR BUILD_WITH_MKL)
  target_compile_definitions(afcpu PRIVATE WITH_LINEAR_ALGEBRA)
endif()

af_split_debug_info(afcpu ${AF_INSTALL_LIB_DIR})

install(TARGETS afcpu
  EXPORT ArrayFireCPUTargets
  COMPONENT cpu
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
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
source_group("" FILES CMakeLists.txt)
