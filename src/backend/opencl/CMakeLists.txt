# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

dependency_check(OpenCL_FOUND "OpenCL not found.")

include(InternalUtils)
include(build_cl2hpp)
include(build_CLBlast)
include(build_clFFT)
include(FileToString)

generate_product_version(af_opencl_ver_res_file
  FILE_NAME "afopencl"
  FILE_DESCRIPTION "OpenCL Backend Dynamic-link library"
)

set(kernel_src
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/KParam.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/anisotropic_diffusion.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx1.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/assign.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bilateral.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve_separable.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/coo2dense.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/copy.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cscmm.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cscmv.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/csr2coo.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/csr2dense.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/csrmm.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/csrmv.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/dense2csr.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diag_create.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diag_extract.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diff.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/example.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fast.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fftconvolve_multiply.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fftconvolve_pack.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fftconvolve_reorder.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/flood_fill.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/gradient.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/harris.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/histogram.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/homography.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/hsv_rgb.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/identity.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iir.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/index.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/interp.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iops.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iota.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ireduce_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ireduce_first.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/jit.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/laset_band.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/laset.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/laswp.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lookup.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lu_split.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/matchTemplate.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/mean_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/mean_first.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/mean_ops.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/meanshift.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/medfilt1.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/medfilt2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/memcopy.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/moments.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/morph.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/nearest_neighbour.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/nonmax_suppression.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ops.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/orb.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/pad_array_borders.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/random_engine_mersenne.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/random_engine_mersenne_init.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/random_engine_philox.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/random_engine_threefry.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/random_engine_write.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/range.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_all.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_blocks_by_key_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_blocks_by_key_first.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_by_key_boundary.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_by_key_boundary_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_by_key_compact.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_by_key_compact_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_by_key_needs_reduction.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reduce_first.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/regions.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reorder.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/resize.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/rotate.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/select.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sift_nonfree.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sobel.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse_arith_common.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse_arith_coo.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse_arith_csr.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sp_sp_arith_csr.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ssarith_calc_out_nnz.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/susan.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/swapdblk.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/tile.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/trace_edge.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transform.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose_inplace.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/triangle.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/unwrap.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/where.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/wrap.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/wrap_dilated.cl
)

set( kernel_headers_dir "kernel_headers")

file_to_string(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${kernel_headers_dir}
    TARGETS cl_kernel_targets
    NAMESPACE "arrayfire opencl"
    )

if(OpenCL_VERSION_MAJOR LESS 3)
  set(opencl_compile_definitions
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS)
else()
  set(opencl_compile_definitions
    CL_TARGET_OPENCL_VERSION=300
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_HPP_MINIMUM_OPENCL_VERSION=110
    CL_HPP_ENABLE_EXCEPTIONS)
endif()

include(kernel/scan_by_key/CMakeLists.txt)
include(kernel/sort_by_key/CMakeLists.txt)

add_library(afopencl "")
add_library(ArrayFire::afopencl ALIAS afopencl)

target_sources(afopencl
  PRIVATE
    $<$<PLATFORM_ID:Windows>:${af_opencl_ver_res_file}>
    Array.cpp
    Array.hpp
    Kernel.cpp
    Kernel.hpp
    Module.hpp
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
    canny.cpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    clfft.cpp
    clfft.hpp
    compile_module.cpp
    complex.hpp
    convolve.cpp
    convolve.hpp
    convolve_separable.cpp
    copy.cpp
    copy.hpp
    count.cpp
    debug_opencl.hpp
    device_manager.cpp
    device_manager.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    err_clblast.hpp
    err_opencl.hpp
    errorcodes.cpp
    errorcodes.hpp
    Event.hpp
    Event.cpp
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
    GraphicsResourceManager.cpp
    GraphicsResourceManager.hpp
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
    resize.cpp
    resize.hpp
    reshape.cpp
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
    threadsMgt.hpp
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
    types.hpp
    types.cpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    vector_field.cpp
    vector_field.hpp
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
    kernel/flood_fill.hpp
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
    kernel/pad_array_borders.hpp
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
    kernel/sift.hpp
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
    jit/BufferNode.hpp
    jit/ShiftNode.hpp
    jit/kernel_generators.hpp
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

if(NOT TARGET clblast)
  add_dependencies(afopencl ${cl_kernel_targets} CLBlast-ext)
endif()

set_target_properties(afopencl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(afopencl
  PRIVATE
    ${opencl_compile_definitions}
    AF_OPENCL
  )

target_link_libraries(afopencl
  PRIVATE
    c_api_interface
    cpp_api_interface
    OpenCL::OpenCL
    OpenCL::cl2hpp
    afcommon_interface
    clFFT
    clblast
    opencl_scan_by_key
    opencl_sort_by_key
    Threads::Threads
    )

if(APPLE)
  target_link_libraries(afopencl PRIVATE OpenGL::GL)
endif()

if(LAPACK_FOUND OR BUILD_WITH_MKL)
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

  if(BUILD_WITH_MKL)
    target_compile_definitions(afopencl PRIVATE USE_MKL)
    if(MKL_BATCH)
      target_compile_definitions(afopencl PRIVATE AF_USE_MKL_BATCH)
    endif()

    if(AF_WITH_STATIC_MKL)
        target_link_libraries(afopencl PRIVATE MKL::Static)
        target_compile_definitions(afopencl PRIVATE USE_STATIC_MKL)
    else()
        target_link_libraries(afopencl PRIVATE MKL::RT)
    endif()
  else()
    if(USE_CPU_F77_BLAS)
      target_compile_definitions(afopencl PRIVATE USE_F77_BLAS)
    endif()

    target_include_directories(afopencl
      SYSTEM PRIVATE
        ${CBLAS_INCLUDE_DIR})

    check_cxx_compiler_flag("-Wl,--start-group -Werror" group_flags)
    if(group_flags)
      set(START_GROUP -Wl,--start-group)
      set(END_GROUP -Wl,--end-group)
    endif()
    target_link_libraries(afopencl
      PRIVATE
        ${START_GROUP}
        ${LAPACK_LIBRARIES}
        LAPACKE::LAPACKE
        ${CBLAS_LIBRARIES}
        ${END_GROUP}
      )
  endif()

  target_compile_definitions(afopencl PRIVATE WITH_LINEAR_ALGEBRA)
endif()

af_split_debug_info(afopencl ${AF_INSTALL_LIB_DIR})

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

if(NOT APPLE AND AF_INSTALL_STANDALONE)
  if(UNIX)
    get_filename_component(opencl_outpath "${OpenCL_LIBRARIES}" REALPATH)
    install(FILES ${opencl_outpath}
        DESTINATION ${AF_INSTALL_LIB_DIR}
        RENAME "${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}.1"
        COMPONENT opencl_dependencies)
  else()
    find_file(OpenCL_DLL_LIBRARY
      NAMES ${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}
	  PATHS
        ENV "PROGRAMFILES(X86)"
        ENV "PROGRAMFILES"
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
        ENV ATISTREAMSDKROOT
      PATH_SUFFIXES
        "AMD APP SDK/bin/x86_64"
        "bin/x86_64"
        "bin/x64"
        "bin/icd/x64"
        "OpenCL SDK/bin/icd/x64"
        "Intel/OpenCL SDK/bin/icd/x64"
        "OpenCL SDK/bin/icd/x64"
        "NVIDIA Corporation/OpenCL")
    mark_as_advanced(OpenCL_DLL_LIBRARY)
    install(FILES "${OpenCL_DLL_LIBRARY}"
        DESTINATION ${AF_INSTALL_BIN_DIR}
        COMPONENT opencl_dependencies)
  endif()
endif()

source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/*)
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/src/backend/build_version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
