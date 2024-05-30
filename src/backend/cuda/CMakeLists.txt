# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

generate_product_version(af_cuda_ver_res_file
  FILE_NAME "afcuda"
  FILE_DESCRIPTION "CUDA Backend Dynamic-link library"
)

dependency_check(CUDA_FOUND "CUDA not found.")
if(AF_WITH_CUDNN)
  dependency_check(cuDNN_FOUND "CUDNN not found.")
endif()

include(AFcuda_helpers)
include(FileToString)
include(InternalUtils)
include(select_compute_arch)

# Remove cublas_device library which is no longer included with the cuda
# toolkit. Fixes issues with older CMake versions
if(DEFINED CUDA_cublas_device_LIBRARY AND NOT CUDA_cublas_device_LIBRARY)
  list(REMOVE_ITEM CUDA_CUBLAS_LIBRARIES ${CUDA_cublas_device_LIBRARY})
endif()

if(NOT OPENGL_FOUND)
  # create a dummy gl.h header to satisfy cuda_gl_interop.h requirement
  # all opengl functionality is made available via glad third party code
  # that is built along with arrayfire code base.
  set(dummy_gl_root "${ArrayFire_BINARY_DIR}/include/GL")
  if(APPLE)
    set(dummy_gl_root "${ArrayFire_BINARY_DIR}/include/OpenGL")
  endif()
  file(WRITE "${dummy_gl_root}/gl.h" "// Dummy file to satisy cuda_gl_interop")
endif()

# Find if CUDA Toolkit is at least 10.0 to use static
# lapack library. Otherwise, we have to use regular shared library
if(UNIX AND (CUDA_VERSION_MAJOR VERSION_GREATER 10 OR CUDA_VERSION_MAJOR VERSION_EQUAL 10))
  set(use_static_cuda_lapack ON)
else()
  set(use_static_cuda_lapack OFF)
endif()

set(CUDA_architecture_build_targets "Auto" CACHE
  STRING "The compute architectures targeted by this build. (Options: Auto;3.0;Maxwell;All;Common)")

find_cuda_helper_libs(nvrtc)
find_cuda_helper_libs(nvrtc-builtins)
list(APPEND nvrtc_libs ${CUDA_nvrtc_LIBRARY})

if(UNIX AND AF_WITH_STATIC_CUDA_NUMERIC_LIBS)
  # The libraries that may be staticly linked or may be loaded at runtime
  set(AF_CUDA_optionally_static_libraries)

  af_multiple_option(NAME        AF_cusparse_LINK_LOADING
    DEFAULT     "Module"
    DESCRIPTION "The approach to load the cusparse library. Static linking(Static) or Dynamic runtime loading(Module) of the module"
    OPTIONS     "Module" "Static")

  if(AF_cusparse_LINK_LOADING STREQUAL "Static")
    af_find_static_cuda_libs(cusparse_static PRUNE)
    list(APPEND AF_CUDA_optionally_static_libraries ${AF_CUDA_cusparse_static_LIBRARY})
  endif()

  af_find_static_cuda_libs(culibos)
  af_find_static_cuda_libs(cublas_static PRUNE)
  af_find_static_cuda_libs(cublasLt_static PRUNE)
  af_find_static_cuda_libs(cufft_static)

  if(CUDA_VERSION VERSION_GREATER 11.4)
    af_find_static_cuda_libs(nvrtc_static)
    af_find_static_cuda_libs(nvrtc-builtins_static)
    af_find_static_cuda_libs(nvptxcompiler_static)
    set(nvrtc_libs ${AF_CUDA_nvrtc_static_LIBRARY}
                   ${AF_CUDA_nvrtc-builtins_static_LIBRARY}
                   ${AF_CUDA_nvptxcompiler_static_LIBRARY})
  endif()

  # FIXME When NVCC resolves this particular issue.
  # NVCC doesn't like -l<full_path_static_lib>, hence we cannot
  # use ${CMAKE_*_LIBRARY} variables in the following flags.
  set(af_cuda_static_flags "${af_cuda_static_flags};-lculibos")
  set(af_cuda_static_flags "${af_cuda_static_flags};-lcublas_static")

  if(CUDA_VERSION VERSION_GREATER 10.0)
    set(af_cuda_static_flags "${af_cuda_static_flags};-lcublasLt_static")
  endif()
  set(af_cuda_static_flags "${af_cuda_static_flags};-lcufft_static")

  if(${use_static_cuda_lapack})
    af_find_static_cuda_libs(cusolver_static PRUNE)
    set(cusolver_static_lib "${AF_CUDA_cusolver_static_LIBRARY}")

    # NVIDIA LAPACK library liblapack_static.a is a subset of LAPACK and only
    # contains GPU accelerated stedc and bdsqr. The user has to link
    # libcusolver_static.a with liblapack_static.a in order to build
    # successfully.
    # Cuda Versions >= 12.0 changed lib name to libcusolver_lapack_static.a
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      af_find_static_cuda_libs(cusolver_lapack_static)
    else()
      af_find_static_cuda_libs(lapack_static)
    endif()

    set(af_cuda_static_flags "${af_cuda_static_flags};-lcusolver_static")
  else()
    set(cusolver_lib "${CUDA_cusolver_LIBRARY}" OpenMP::OpenMP_CXX)
  endif()
endif()

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_cudart_static_LIBRARY} DIRECTORY CACHE)

mark_as_advanced(
    CUDA_LIBRARIES_PATH
    CUDA_architecture_build_targets)

if(CUDA_VERSION_MAJOR VERSION_LESS 11)
  find_package(CUB)
  if(NOT TARGET CUB::CUB)
    af_dep_check_and_populate(${cub_prefix}
      URI https://github.com/NVIDIA/cub.git
      REF 1.10.0
    )
    find_package(CUB REQUIRED
      PATHS ${${cub_prefix}_SOURCE_DIR})
  endif()
endif()

file(GLOB jit_src "kernel/jit.cuh")

file_to_string(
    SOURCES ${jit_src}
    VARNAME jit_files
    EXTENSION "hpp"
    OUTPUT_DIR "kernel_headers"
    TARGETS jit_kernel_targets
    NAMESPACE "arrayfire cuda"
    WITH_EXTENSION
    )

set(nvrtc_src
  ${CUDA_INCLUDE_DIRS}/cuda_fp16.h
  ${CUDA_INCLUDE_DIRS}/cuda_fp16.hpp
  ${CUDA_TOOLKIT_ROOT_DIR}/include/cuComplex.h
  ${CUDA_TOOLKIT_ROOT_DIR}/include/math_constants.h
  ${CUDA_TOOLKIT_ROOT_DIR}/include/vector_types.h
  ${CUDA_TOOLKIT_ROOT_DIR}/include/vector_functions.h

  ${PROJECT_SOURCE_DIR}/src/api/c/optypes.hpp
  ${PROJECT_SOURCE_DIR}/include/af/defines.h
  ${PROJECT_SOURCE_DIR}/include/af/traits.hpp
  ${PROJECT_BINARY_DIR}/include/af/version.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assign_kernel_param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dims_param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/interp.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/shared.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/minmax_op.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utility.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/Binary.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/Transform.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/half.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/internal_enums.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/kernel_type.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/anisotropic_diffusion.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx1.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx2.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/assign.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bilateral.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/canny.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve1.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve2.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve3.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve_separable.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/copy.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diagonal.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diff.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/exampleFunction.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fftconvolve.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/flood_fill.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/gradient.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/histogram.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/hsv_rgb.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/identity.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iir.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/index.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iota.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ireduce.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lookup.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lu_split.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/match_template.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/meanshift.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/medfilt.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/memcopy.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/moments.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/morph.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/pad_array_borders.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/range.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/resize.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reorder.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/rotate.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/select.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sobel.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse_arith.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/susan.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/tile.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transform.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose_inplace.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/triangle.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/unwrap.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/where.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/wrap.cuh
  )

file_to_string(
    SOURCES ${nvrtc_src}
    VARNAME nvrtc_files
    EXTENSION "hpp"
    OUTPUT_DIR "nvrtc_kernel_headers"
    TARGETS nvrtc_kernel_targets
    NAMESPACE "arrayfire cuda"
    WITH_EXTENSION
    NULLTERM
    )

include(kernel/scan_by_key/CMakeLists.txt)
include(kernel/thrust_sort_by_key/CMakeLists.txt)


add_library(afcuda
    $<$<PLATFORM_ID:Windows>:${af_cuda_ver_res_file}>
    ${thrust_sort_sources}

    blas.cu
    blas.hpp
    cudaDataType.hpp
    cufft.cu
    cufft.hpp
    cusparse_descriptor_helpers.hpp
    fft.cu
    sparse.cu
    sparse.hpp
    sparse_arith.cu
    sparse_arith.hpp
    sparse_blas.cu
    sparse_blas.hpp
    solve.cu
    solve.hpp

    EnqueueArgs.hpp
    all.cu
    anisotropic_diffusion.cpp
    any.cu
    approx.cpp
    bilateral.cpp
    canny.cpp
    count.cu
    Event.cpp
    Event.hpp
    exampleFunction.cpp
    fast.cu
    harris.cu
    histogram.cpp
    homography.cu
    hsv_rgb.cpp
    match_template.cpp
    max.cu
    mean.cu
    meanshift.cpp
    medfilt.cpp
    min.cu
    moments.cpp
    nearest_neighbour.cu
    orb.cu
    pad_array_borders.cpp
    product.cu
    random_engine.cu
    regions.cu
    resize.cpp
    rotate.cpp
    set.cu
    sift.cu
    sobel.cpp
    sort.cu
    sort_by_key.cu
    sort_index.cu
    sum.cu
    topk.cu
    transform.cpp
    transpose.cpp
    transpose_inplace.cpp

    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/atomics.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/config.hpp
    kernel/convolve.hpp
    kernel/convolve_separable.cpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fast_lut.hpp
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
    kernel/lookup.hpp
    kernel/lu_split.hpp
    kernel/match_template.hpp
    kernel/mean.hpp
    kernel/meanshift.hpp
    kernel/medfilt.hpp
    kernel/memcopy.hpp
    kernel/moments.hpp
    kernel/morph.hpp
    kernel/nearest_neighbour.hpp
    kernel/orb.hpp
    kernel/orb_patch.hpp
    kernel/pad_array_borders.hpp
    kernel/random_engine.hpp
    kernel/random_engine_mersenne.hpp
    kernel/random_engine_philox.hpp
    kernel/random_engine_threefry.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/reduce_by_key.hpp
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
    kernel/shared.hpp
    kernel/shfl_intrinsics.hpp
    kernel/sift.hpp
    kernel/sobel.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sparse.hpp
    kernel/sparse_arith.hpp
    kernel/susan.hpp
    kernel/thrust_sort_by_key.hpp
    kernel/thrust_sort_by_key_impl.hpp
    kernel/tile.hpp
    kernel/topk.hpp
    kernel/transform.hpp
    kernel/transpose.hpp
    kernel/transpose_inplace.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/where.hpp
    kernel/wrap.hpp

    Array.cpp
    Array.hpp
    Kernel.cpp
    Kernel.hpp
    LookupTable1D.hpp
    Module.hpp
    Param.hpp
    ThrustAllocator.cuh
    ThrustArrayFirePolicy.hpp
    anisotropic_diffusion.hpp
    approx.hpp
    arith.hpp
    assign.cpp
    assign.hpp
    backend.hpp
    bilateral.hpp
    binary.hpp
    blas.hpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    complex.hpp
    compile_module.cpp
    convolve.cpp
    convolve.hpp
    convolveNN.cpp
    copy.cpp
    copy.hpp
    cublas.cpp
    cublas.hpp

    $<$<BOOL:${AF_WITH_CUDNN}>: cudnn.cpp
                             cudnn.hpp
                             cudnnModule.cpp
                             cudnnModule.hpp>

    cufft.hpp
    cusolverDn.cpp
    cusolverDn.hpp
    cusparse.cpp
    cusparse.hpp
    cusparseModule.cpp
    cusparseModule.hpp
    device_manager.cpp
    device_manager.hpp
    debug_cuda.hpp
    thrust_utils.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    driver.cpp
    err_cuda.hpp
    exampleFunction.hpp
    fast.hpp
    fast_pyramid.cpp
    fast_pyramid.hpp
    fft.hpp
    fftconvolve.cpp
    fftconvolve.hpp
    flood_fill.cpp
    flood_fill.hpp
    GraphicsResourceManager.cpp
    GraphicsResourceManager.hpp
    gradient.cpp
    gradient.hpp
    harris.hpp
    hist_graphics.cpp
    hist_graphics.hpp
    histogram.hpp
    homography.hpp
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
    match_template.hpp
    math.hpp
    mean.hpp
    meanshift.hpp
    medfilt.hpp
    memory.cpp
    memory.hpp
    minmax_op.hpp
    moments.hpp
    morph.cpp
    morph.hpp
    nearest_neighbour.hpp
    orb.hpp
    platform.cpp
    platform.hpp
    plot.cpp
    plot.hpp
    print.hpp
    qr.cpp
    qr.hpp
    random_engine.hpp
    range.cpp
    range.hpp
    reduce.hpp
    reduce_impl.hpp
    regions.hpp
    reorder.cpp
    reorder.hpp
    resize.hpp
    reshape.cpp
    rotate.hpp
    scalar.hpp
    scan.cpp
    scan.hpp
    scan_by_key.cpp
    scan_by_key.hpp
    select.cpp
    select.hpp
    set.hpp
    shift.cpp
    shift.hpp
    sift.hpp
    sobel.hpp
    solve.hpp
    sort.hpp
    sort_by_key.hpp
    sort_index.hpp
    sparse.hpp
    sparse_arith.hpp
    sparse_blas.hpp
    surface.cpp
    surface.hpp
    susan.cpp
    susan.hpp
    svd.cpp
    svd.hpp
    tile.cpp
    tile.hpp
    threadsMgt.hpp
    topk.hpp
    traits.hpp
    transform.hpp
    transpose.hpp
    triangle.cpp
    triangle.hpp
    types.hpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    utility.cpp
    utility.hpp
    vector_field.cpp
    vector_field.hpp
    where.cpp
    where.hpp
    wrap.cpp
    wrap.hpp

    jit/BufferNode.hpp
    jit/ShiftNode.hpp
    jit/kernel_generators.hpp

    ${scan_by_key_sources}
  )


if(UNIX AND AF_WITH_STATIC_CUDA_NUMERIC_LIBS)
  check_cxx_compiler_flag("-Wl,--start-group -Werror" group_flags)
  if(group_flags)
    set(START_GROUP -Wl,--start-group)
    set(END_GROUP -Wl,--end-group)
  endif()

  target_link_libraries(afcuda
    PRIVATE
      ${cusolver_lib}
      ${START_GROUP}
      ${CUDA_culibos_LIBRARY} #also a static libary
      ${AF_CUDA_cublas_static_LIBRARY}
      ${AF_CUDA_cublasLt_static_LIBRARY}
      ${AF_CUDA_cufft_static_LIBRARY}
      ${AF_CUDA_optionally_static_libraries}
      ${nvrtc_libs}
      ${cusolver_static_lib}
      ${END_GROUP})

  if(CUDA_VERSION VERSION_GREATER 10.0)
    target_link_libraries(afcuda
      PRIVATE
        ${AF_CUDA_cublasLt_static_LIBRARY})
  endif()

  if(CUDA_VERSION VERSION_GREATER 9.5)
    target_link_libraries(afcuda
      PRIVATE
        ${CUDA_lapack_static_LIBRARY})
  endif()

else()
  target_link_libraries(afcuda
    PRIVATE
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_cusolver_LIBRARY}
      ${nvrtc_libs}
  )
endif()

if(CUDA_VERSION_MAJOR VERSION_LESS 11)
  target_link_libraries(afcuda
    PRIVATE
      CUB::CUB
  )
endif()

af_detect_and_set_cuda_architectures(afcuda)

if(CUDA_VERSION VERSION_LESS 11.0)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_target_properties(afcuda
      PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON)
  else()
    target_compile_options(afcuda
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--std=c++14>)
  endif()
else()
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_target_properties(afcuda
      PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
  else()
    target_compile_options(afcuda
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
  endif()
endif()

target_compile_definitions(afcuda
  PRIVATE
    AF_CUDA

    # CUDA_NO_HALF prevents the inclusion of the half class in the global namespace
    # which conflicts with the half class in ArrayFire's common namespace. prefer
    # using __half class instead for CUDA
    CUDA_NO_HALF

    $<$<BOOL:${AF_WITH_CUDNN}>:WITH_CUDNN>
)

# New API of cuSparse was introduced in 10.1.168 for Linux and the older
# 10.1.105 fix version doesn't it. Unfortunately, the new API was introduced in
# in a fix release of CUDA - unconventionally. As CMake's FindCUDA module
# doesn't provide patch/fix version number, we use 10.2 as the minimum
# CUDA version to enable this new cuSparse API.
if(CUDA_VERSION_MAJOR VERSION_GREATER 10 OR
    (UNIX AND
      CUDA_VERSION_MAJOR VERSION_EQUAL 10 AND CUDA_VERSION_MINOR VERSION_GREATER 1))
  target_compile_definitions(afcuda
    PRIVATE
      AF_USE_NEW_CUSPARSE_API)
endif()

target_compile_options(afcuda
  PRIVATE
    $<$<BOOL:${AF_WITH_FAST_MATH}>:$<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=unrecognized_gcc_pragma>
    $<$<COMPILE_LANGUAGE:CUDA>: $<$<CXX_COMPILER_ID:MSVC>:  -Xcompiler=/wd4251
                                                            -Xcompiler=/wd4068
                                                            -Xcompiler=/wd4275
                                                            -Xcompiler=/wd4668
                                                            -Xcompiler=/wd4710
                                                            -Xcompiler=/wd4505
                                                            -Xcompiler=/bigobj>>
)


if(UNIX AND AF_WITH_STATIC_CUDA_NUMERIC_LIBS AND AF_cusparse_LINK_LOADING STREQUAL "Static")
  target_compile_definitions(afcuda
    PRIVATE
      AF_cusparse_STATIC_LINKING)
endif()

add_library(ArrayFire::afcuda ALIAS afcuda)

add_dependencies(afcuda ${jit_kernel_targets} ${nvrtc_kernel_targets})

if(UNIX AND AF_WITH_PRUNE_STATIC_CUDA_NUMERIC_LIBS)
  add_dependencies(afcuda ${cuda_pruned_library_targets})
endif()

target_include_directories (afcuda
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${ArrayFire_SOURCE_DIR}/src/api/c
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/jit
    ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories (afcuda
  SYSTEM PRIVATE
    $<$<BOOL:${AF_WITH_CUDNN}>:${cuDNN_INCLUDE_DIRS}>
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(afcuda
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    ${CMAKE_DL_LIBS}
  )

# If the driver is not found the cuda driver api need to be linked against the
# libcuda.so stub located in the lib[64]/stubs directory
if(CUDA_CUDA_LIBRARY)
  target_link_libraries(afcuda PRIVATE ${CUDA_CUDA_LIBRARY})
else()
  message(STATUS "CUDA driver library missing. Looking for libcuda stub.")
  find_library(CUDA_CUDA_STUB
             NAMES cuda
             PATHS ${CUDA_LIBRARIES_PATH}/stubs
             NO_DEFAULT_PATH
         )
  if(CUDA_CUDA_STUB)
    message(STATUS "CUDA driver stub FOUND: ${CUDA_CUDA_STUB}")
  endif()

  #NOTE: Only link against the stub library when building
  target_link_libraries(afcuda
    PUBLIC
      $<BUILD_INTERFACE:${CUDA_CUDA_STUB}>)
endif()

# TODO(umar): This is required for NVRTC to work correctly on OSX. It may not
#             be necessary on other platforms.
if(APPLE)
  target_link_libraries(afcuda PUBLIC -Wl,-rpath,${CUDA_LIBRARIES_PATH})
endif()

af_split_debug_info(afcuda ${AF_INSTALL_LIB_DIR})

install(TARGETS afcuda
  EXPORT ArrayFireCUDATargets
  COMPONENT cuda
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR}
  )

set(cuda_deps "")
set (PX ${CMAKE_SHARED_LIBRARY_PREFIX})
set (SX ${CMAKE_SHARED_LIBRARY_SUFFIX})
set (dlib_path_prefix ${CUDA_LIBRARIES_PATH})
if (WIN32)
  set(dlib_path_prefix "${CUDA_TOOLKIT_ROOT_DIR}/bin")
endif ()

function(afcu_collect_libs libname)
  set(options "FULL_VERSION")
  set(single_args "LIB_MAJOR;LIB_MINOR")
  set(multi_args "")

  cmake_parse_arguments(cuda_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  if(cuda_args_LIB_MAJOR AND cuda_args_LIB_MINOR)
    set(lib_major ${cuda_args_LIB_MAJOR})
	  set(lib_minor ${cuda_args_LIB_MINOR})
  else()
    set(lib_major ${CUDA_VERSION_MAJOR})
	  set(lib_minor ${CUDA_VERSION_MINOR})
  endif()
  set(lib_version "${lib_major}.${lib_minor}")

  if (WIN32)
    find_file(CUDA_${libname}_LIBRARY_DLL
      NAMES
        "${PX}${libname}64_${lib_major}${SX}"
        "${PX}${libname}64_${lib_major}${lib_minor}${SX}"
        "${PX}${libname}64_${lib_major}${lib_minor}_0${SX}"
      PATHS ${dlib_path_prefix}
    )
    mark_as_advanced(CUDA_${libname}_LIBRARY_DLL)
    install(FILES "${CUDA_${libname}_LIBRARY_DLL}"
      DESTINATION ${AF_INSTALL_BIN_DIR}
      COMPONENT cuda_dependencies)
  elseif (APPLE)
    get_filename_component(outpath "${dlib_path_prefix}/${PX}${libname}.${lib_major}.${lib_minor}${SX}" REALPATH)
    install(FILES       "${outpath}"
            DESTINATION ${AF_INSTALL_BIN_DIR}
            RENAME      "${PX}${libname}.${lib_version}${SX}"
            COMPONENT   cuda_dependencies)
  else () #UNIX
    find_library(CUDA_${libname}_LIBRARY
      NAMES ${libname}
      PATHS
        ${dlib_path_prefix})

    get_filename_component(outpath "${CUDA_${libname}_LIBRARY}" REALPATH)
    if(cuda_args_FULL_VERSION)
      set(library_install_name "${PX}${libname}${SX}.${lib_version}")
    else()
      set(library_install_name "${PX}${libname}${SX}.${lib_major}")
    endif()
    install(FILES       ${outpath}
            DESTINATION ${AF_INSTALL_LIB_DIR}
            RENAME      ${library_install_name}
            COMPONENT   cuda_dependencies)
  endif ()
endfunction()

function(afcu_collect_cudnn_libs cudnn_infix)
  set(internal_infix "_")
  if(NOT "${cudnn_infix}" STREQUAL "")
    set(internal_infix "_${cudnn_infix}_")
    string(TOUPPER ${internal_infix} internal_infix)
  endif()
  if(WIN32)
    set(cudnn_lib "${cuDNN${internal_infix}DLL_LIBRARY}")
  else()
    get_filename_component(cudnn_lib "${cuDNN${internal_infix}LINK_LIBRARY}" REALPATH)
  endif()
  install(FILES ${cudnn_lib} DESTINATION ${AF_INSTALL_LIB_DIR} COMPONENT cuda_dependencies)
endfunction()

if(AF_INSTALL_STANDALONE)
  if(AF_WITH_CUDNN)
    afcu_collect_cudnn_libs("")
    if(cuDNN_VERSION_MAJOR VERSION_GREATER 8 OR cuDNN_VERSION_MAJOR VERSION_EQUAL 8)
      # cudnn changed how dlls are shipped starting major version 8
      # except the main dll a lot of the other DLLs are loaded upon demand
      afcu_collect_cudnn_libs(cnn_infer)
      afcu_collect_cudnn_libs(cnn_train)
      afcu_collect_cudnn_libs(ops_infer)
      afcu_collect_cudnn_libs(ops_train)
    endif()
  endif()

  if(WIN32 OR NOT AF_WITH_STATIC_CUDA_NUMERIC_LIBS)
    if(CUDA_VERSION_MAJOR VERSION_EQUAL 11)
        afcu_collect_libs(cufft LIB_MAJOR 10 LIB_MINOR 4)
    else()
        afcu_collect_libs(cufft)
    endif()
    afcu_collect_libs(cublas)
    if(CUDA_VERSION VERSION_GREATER 10.0)
      afcu_collect_libs(cublasLt)
    endif()
    afcu_collect_libs(cusolver)
    afcu_collect_libs(cusparse)
  elseif(NOT ${use_static_cuda_lapack})
    afcu_collect_libs(cusolver)
  endif()

  if(WIN32 OR CUDA_VERSION VERSION_LESS 11.5 OR NOT AF_WITH_STATIC_CUDA_NUMERIC_LIBS)
    afcu_collect_libs(nvrtc FULL_VERSION)
    if(CUDA_VERSION VERSION_GREATER 10.0)
      afcu_collect_libs(nvrtc-builtins FULL_VERSION)
    else()
      if(APPLE)
        afcu_collect_libs(cudart)

        get_filename_component(nvrtc_outpath "${dlib_path_prefix}/${PX}nvrtc-builtins.${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}${SX}" REALPATH)
        install(FILES       ${nvrtc_outpath}
                DESTINATION ${AF_INSTALL_BIN_DIR}
                RENAME      "${PX}nvrtc-builtins${SX}"
                COMPONENT   cuda_dependencies)
      elseif(UNIX)
        get_filename_component(nvrtc_outpath "${dlib_path_prefix}/${PX}nvrtc-builtins${SX}" REALPATH)
        install(FILES       ${nvrtc_outpath}
                DESTINATION ${AF_INSTALL_LIB_DIR}
                RENAME      "${PX}nvrtc-builtins${SX}"
                COMPONENT   cuda_dependencies)
      else()
        afcu_collect_libs(nvrtc-builtins)
      endif()
    endif()
  endif()
endif()


source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/*)
source_group("generated files"  FILES ${ArrayFire_BINARY_DIR}/src/backend/build_version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h
                                REGULAR_EXPRESSION ${CMAKE_CURRENT_BINARY_DIR}/${kernel_headers_dir}/*)
source_group("" FILES CMakeLists.txt)

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_NV_CUB
  FETCHCONTENT_UPDATES_DISCONNECTED_NV_CUB
)
