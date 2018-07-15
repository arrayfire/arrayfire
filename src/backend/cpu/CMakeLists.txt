# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)

add_library(afcpu "")
add_library(ArrayFire::afcpu ALIAS afcpu)

# CPU backend source files
target_sources(afcpu
  PRIVATE
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
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    err_cpu.hpp
    exampleFunction.cpp
    exampleFunction.hpp
    fast.cpp
    fast.hpp
    fft.cpp
    fft.hpp
    fftconvolve.cpp
    fftconvolve.hpp
    gradient.cpp
    gradient.hpp
    harris.cpp
    harris.hpp
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
    padarray.cpp
    platform.cpp
    platform.hpp
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
    kernel/fft.hpp
    kernel/fftconvolve.hpp
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

target_sources(afcpu
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/threads/async_queue.hpp
  )

if(MSVC)
  target_compile_options(afcpu PRIVATE /bigobj)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/CMakeLists.txt")

if(AF_WITH_NONFREE)
  target_sources(afcpu PRIVATE kernel/sift_nonfree.hpp)
  target_compile_definitions(afcpu PRIVATE AF_WITH_NONFREE_SIFT)
endif()

if(AF_WITH_GRAPHICS)
  if(NOT AF_USE_SYSTEM_FORGE)
    add_dependencies(afcpu forge-ext)
  endif()
  target_sources(afcpu
    PRIVATE
      hist_graphics.cpp
      hist_graphics.hpp
      image.cpp
      image.hpp
      plot.cpp
      plot.hpp
      surface.cpp
      surface.hpp
      vector_field.cpp
      vector_field.hpp
    )
endif()

target_include_directories(afcpu
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    threads
    ${CBLAS_INCLUDE_DIR}
  )

# TODO(umar) Find a better way to determine BLAS selection
if(USE_CPU_MKL)
  dependency_check(MKL_FOUND "MKL not found")
  target_compile_definitions(afcpu PRIVATE USE_MKL)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND (UNIX AND NOT APPLE)))
    # MKL requires multiple passes when linking with the static libs. This can be
    # done in CMake using LINK_INTERFACE_MULTIPLICITY but that will require
    # changine the way FindCBLAS works. This can also be done using the
    # --start-group and --end-group linker around the libraries in the linking
    # step.
    #
    # TODO(umar): Change the way CBLAS libraries are found and linked
    set(CBLAS_LIBRARIES -Wl,--start-group ${CBLAS_LIBRARIES} -Wl,--end-group)
  endif()
endif()

target_compile_definitions(afcpu
  PRIVATE
    AF_CPU
  )

if(USE_CPU_MKL)
  target_link_libraries(afcpu
    PRIVATE
      c_api_interface
      cpp_api_interface
      afcommon_interface
      cpu_sort_by_key
      MKL::MKL
      Threads::Threads
    )
else()
  dependency_check(FFTW_FOUND "FFTW not found")
  dependency_check(CBLAS_FOUND "CBLAS not found")

  target_link_libraries(afcpu
    PRIVATE
      c_api_interface
      cpp_api_interface
      afcommon_interface
      cpu_sort_by_key
      ${CBLAS_LIBRARIES}
      FFTW::FFTW
      FFTW::FFTWF
      Threads::Threads
    )
  if(LAPACK_FOUND)
    target_link_libraries(afcpu
      PRIVATE
        ${LAPACK_LIBRARIES})
    target_include_directories(afcpu
      PRIVATE
        ${LAPACK_INCLUDE_DIR})
  endif()
endif()

if(LAPACK_FOUND OR MKL_FOUND)
  target_compile_definitions(afcpu
    PRIVATE
      WITH_LINEAR_ALGEBRA)
endif()

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
