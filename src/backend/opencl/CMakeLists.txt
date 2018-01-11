# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)

set(OPENCL_BLAS_LIBRARY clBLAS CACHE STRING "Select OpenCL BLAS back-end")
set_property(CACHE OPENCL_BLAS_LIBRARY PROPERTY STRINGS "clBLAS" "CLBlast")

include(build_clFFT)

file(GLOB kernel_src kernel/*.cl kernel/KParam.hpp)

set( kernel_headers_dir
    "kernel_headers")

include(CLKernelToH)

cl_kernel_to_h(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${kernel_headers_dir}
    TARGETS cl_kernel_targets
    NAMESPACE "opencl"
    )

include(kernel/scan_by_key/CMakeLists.txt)
include(kernel/sort_by_key/CMakeLists.txt)

add_library(afopencl "")
add_library(ArrayFire::afopencl ALIAS afopencl)

target_sources(afopencl
  PRIVATE
    Array.cpp
    Array.hpp
    Param.cpp
    Param.hpp
    all.cpp
    anisotropic_diffusion.cpp
    anisotropic_diffusion.hpp
    any.cpp
    api.cpp
    approx.cpp
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
    cache.hpp
    canny.cpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    clfft.cpp
    clfft.hpp
    complex.hpp
    convolve.cpp
    convolve.hpp
    convolve_separable.cpp
    copy.cpp
    copy.hpp
    count.cpp
    debug_opencl.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    dilate.cpp
    dilate3d.cpp
    erode.cpp
    erode3d.cpp
    err_clblas.hpp
    err_clblast.hpp
    err_opencl.hpp
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
    jit.cpp
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
    moments.cpp
    moments.hpp
    morph.hpp
    morph3d_impl.hpp
    morph_impl.hpp
    nearest_neighbour.cpp
    nearest_neighbour.hpp
    orb.cpp
    orb.hpp
    platform.cpp
    platform.hpp
    print.hpp
    product.cpp
    program.cpp
    program.hpp
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
    transpose_inplace.cpp
    triangle.cpp
    triangle.hpp
    types.cpp
    types.hpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    where.cpp
    where.hpp
    wrap.cpp
    wrap.hpp
    )


target_sources(afopencl
  PRIVATE
    kernel/KParam.hpp
    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/config.cpp
    kernel/config.hpp
    kernel/convolve.hpp
    kernel/convolve_separable.cpp
    kernel/convolve_separable.hpp
    kernel/cscmm.hpp
    kernel/cscmv.hpp
    kernel/csrmm.hpp
    kernel/csrmv.hpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fftconvolve.hpp
    kernel/gradient.hpp
    kernel/harris.hpp
    kernel/histogram.hpp
    kernel/homography.hpp
    kernel/hsv_rgb.hpp
    kernel/identity.hpp
    kernel/iir.hpp
    kernel/index.hpp
    kernel/interp.hpp
    kernel/iota.hpp
    kernel/ireduce.hpp
    kernel/join.hpp
    kernel/laset.hpp
    #kernel/laset_band.hpp
    kernel/laswp.hpp
    kernel/lookup.hpp
    kernel/lu_split.hpp
    kernel/match_template.hpp
    kernel/mean.hpp
    kernel/meanshift.hpp
    kernel/medfilt.hpp
    kernel/memcopy.hpp
    kernel/moments.hpp
    kernel/morph.hpp
    kernel/names.hpp
    kernel/nearest_neighbour.hpp
    kernel/orb.hpp
    kernel/random_engine.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/regions.hpp
    kernel/reorder.hpp
    kernel/resize.hpp
    kernel/rotate.hpp
    kernel/scan_dim.hpp
    kernel/scan_dim_by_key.hpp
    kernel/scan_dim_by_key_impl.hpp
    kernel/scan_first.hpp
    kernel/scan_first_by_key.hpp
    kernel/scan_first_by_key_impl.hpp
    kernel/select.hpp
    kernel/sobel.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sort_by_key_impl.hpp
    kernel/sort_helper.hpp
    kernel/sparse.hpp
    kernel/sparse_arith.hpp
    kernel/susan.hpp
    kernel/swapdblk.hpp
    kernel/tile.hpp
    kernel/transform.hpp
    kernel/transpose.hpp
    kernel/transpose_inplace.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/where.hpp
    kernel/wrap.hpp

    kernel/convolve/conv1.cpp
    kernel/convolve/conv2_b8.cpp
    kernel/convolve/conv2_c32.cpp
    kernel/convolve/conv2_c64.cpp
    kernel/convolve/conv2_f32.cpp
    kernel/convolve/conv2_f64.cpp
    kernel/convolve/conv2_impl.hpp
    kernel/convolve/conv2_s16.cpp
    kernel/convolve/conv2_s32.cpp
    kernel/convolve/conv2_s64.cpp
    kernel/convolve/conv2_u16.cpp
    kernel/convolve/conv2_u32.cpp
    kernel/convolve/conv2_u64.cpp
    kernel/convolve/conv2_u8.cpp
    kernel/convolve/conv3.cpp
    kernel/convolve/conv_common.hpp
    )

target_sources(afopencl
  PRIVATE
    JIT/BinaryNode.hpp
    JIT/BufferNode.hpp
    JIT/Node.hpp
    JIT/ScalarNode.hpp
    JIT/UnaryNode.hpp
  )

target_sources(afopencl
  PRIVATE
    ${kernel_files}
  )

target_sources(afopencl
  PRIVATE
    cpu/cpu_blas.cpp
    cpu/cpu_blas.hpp
    cpu/cpu_cholesky.cpp
    cpu/cpu_cholesky.hpp
    cpu/cpu_helper.hpp
    cpu/cpu_inverse.cpp
    cpu/cpu_inverse.hpp
    cpu/cpu_lu.cpp
    cpu/cpu_lu.hpp
    cpu/cpu_qr.cpp
    cpu/cpu_qr.hpp
    cpu/cpu_solve.cpp
    cpu/cpu_solve.hpp
    cpu/cpu_sparse_blas.cpp
    cpu/cpu_sparse_blas.hpp
    cpu/cpu_svd.cpp
    cpu/cpu_svd.hpp
    cpu/cpu_triangle.hpp
  )

target_include_directories(afopencl
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    magma
    ../../api/c
    ../../../include
  )

add_dependencies(afopencl ${cl_kernel_targets})
add_dependencies(opencl_scan_by_key ${cl_kernel_targets} cl2hpp Boost::boost)
add_dependencies(opencl_sort_by_key ${cl_kernel_targets} cl2hpp Boost::boost)

set_target_properties(afopencl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(afopencl
  PRIVATE
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
    __CL_ENABLE_EXCEPTIONS
  )

target_link_libraries(afopencl
  PRIVATE
    c_api_interface
    cpp_api_interface
    OpenCL::OpenCL
    OpenCL::cl2hpp
    afcommon_interface
    clFFT::clFFT
    opencl_scan_by_key
    opencl_sort_by_key
    Boost::boost
    Threads::Threads
    )

if(OPENCL_BLAS_LIBRARY STREQUAL "clBLAS")
  include(build_clBLAS)
  target_compile_definitions(afopencl PRIVATE USE_CLBLAS)
  target_link_libraries(afopencl
    PRIVATE
      clBLAS::clBLAS)
elseif(OPENCL_BLAS_LIBRARY STREQUAL "CLBlast")
  include(build_CLBlast)
  target_compile_definitions(afopencl PRIVATE USE_CLBLAST)
  target_link_libraries(afopencl
    PRIVATE
      CLBlast)
    add_dependencies(afopencl CLBlast-ext)
endif()


if(AF_WITH_NONFREE)
  target_sources(afopencl PRIVATE kernel/sift_nonfree.hpp)
  target_compile_definitions(afopencl PRIVATE AF_WITH_NONFREE_SIFT)
endif()

if(AF_WITH_GRAPHICS)
  target_sources(afopencl
    PRIVATE
      GraphicsResourceManager.hpp
      GraphicsResourceManager.cpp
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

if(LAPACK_FOUND)
  target_sources(afopencl
    PRIVATE
      magma/gebrd.cpp
      magma/geqrf2.cpp
      magma/geqrf3.cpp
      magma/getrf.cpp
      magma/getrs.cpp
      magma/labrd.cpp
      magma/larfb.cpp
      magma/laset.cpp
      #magma/laset_band.cpp
      magma/laswp.cpp
      magma/magma.h
      magma/magma_blas.h
      magma/magma_blas_clblas.h
      magma/magma_blas_clblast.h
      magma/magma_common.h
      magma/magma_cpu_blas.h
      magma/magma_cpu_lapack.h
      magma/magma_data.h
      magma/magma_helper.cpp
      magma/magma_helper.h
      magma/magma_sync.h
      magma/magma_types.h
      magma/potrf.cpp
      magma/swapdblk.cpp
      magma/transpose.cpp
      magma/transpose_inplace.cpp
      magma/ungqr.cpp
      magma/unmqr.cpp
      #magma/unmqr2.cpp

    )

  if(USE_OPENCL_MKL)
    dependency_check(MKL_FOUND "MKL not found")
    target_compile_definitions(afopencl PRIVATE USE_MKL)

    target_link_libraries(afopencl
      PRIVATE
        MKL::MKL
      )

  else()
    dependency_check(OpenCL_FOUND "OpenCL not found.")

    if(USE_CPU_F77_BLAS)
      target_compile_definitions(afopencl PRIVATE USE_F77_BLAS)
    endif()

    dependency_check(CBLAS_LIBRARIES "CBLAS not found.")
    target_include_directories(afopencl
      PRIVATE
        ${CBLAS_INCLUDE_DIR})
    target_link_libraries(afopencl
      PRIVATE
        ${CBLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})
  endif()

  target_compile_definitions(
    afopencl
    PRIVATE
      WITH_OPENCL_LINEAR_ALGEBRA
    )

  target_link_libraries(afopencl
    PRIVATE
      afcommon_lapack_interface)
endif(LAPACK_FOUND)

install(TARGETS afopencl
  EXPORT ArrayFireOpenCLTargets
  COMPONENT opencl
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR}
  )

get_prerequisites("${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}afopencl${CMAKE_SHARED_LIBRARY_SUFFIX}" opencl_deps 1 0 0 "" "")

install(FILES ${opencl_deps}
        DESTINATION ${AF_INSTALL_LIB_DIR}
        COMPONENT   opencl_dependencies)


source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/*)
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
