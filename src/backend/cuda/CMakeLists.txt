# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)
include(select_compute_arch)

dependency_check(CUDA_FOUND "CUDA not found.")

find_cuda_helper_libs(nvrtc)
find_cuda_helper_libs(nvrtc-builtins)

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_cudart_static_LIBRARY} DIRECTORY CACHE)

include(CLKernelToH)

if(NOT CUDA_architecture_build_targets)
  cuda_detect_installed_gpus(detected_gpus)
endif()

set(CUDA_architecture_build_targets ${detected_gpus} CACHE
  STRING "The compute architectures targeted by this build. (Options: 3.0;Maxwell;All;Common)")

cuda_select_nvcc_arch_flags(cuda_architecture_flags ${CUDA_architecture_build_targets})
message(STATUS "CUDA_architecture_build_targets: ${CUDA_architecture_build_targets}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  ${cuda_architecture_flags}
  )

mark_as_advanced(
    CUDA_LIBRARIES_PATH
    CUDA_architecture_build_targets)

cuda_include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ArrayFire_SOURCE_DIR}/include
  ${ArrayFire_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/jit
  ${CMAKE_CURRENT_SOURCE_DIR}/cub
  ${ArrayFire_SOURCE_DIR}/src/api/c
  ${ArrayFire_SOURCE_DIR}/src/backend

  # NOTE: Space after comma is necessary
  $<JOIN:$<TARGET_PROPERTY:afcommon_interface,INTERFACE_INCLUDE_DIRECTORIES>, >
  )

set(jit_kernel_headers
    "kernel_headers")

file(GLOB jit_src "kernel/jit.cuh")

cl_kernel_to_h(
    SOURCES ${jit_src}
    VARNAME jit_files
    EXTENSION "hpp"
    OUTPUT_DIR ${jit_kernel_headers}
    TARGETS jit_kernel_targets
    NAMESPACE "cuda"
    )

## Copied from FindCUDA.cmake
## The target_link_library needs to link with the cuda libraries using
## PRIVATE
function(cuda_add_library cuda_target)
  cuda_add_cuda_include_once()

  # Separate the sources from the options
  cuda_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_build_shared_library(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  cuda_wrap_srcs( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  cuda_compute_separable_compilation_object_file_name(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  cuda_link_separable_compilation_objects("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
      PRIVATE ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endfunction()

arrayfire_get_cuda_cxx_flags(cuda_cxx_flags)

if(AF_WITH_NONFREE AND CMAKE_VERSION VERSION_LESS "3.7")
  # This definition is required in addition to the definition below because in
  # an older verion of cmake definitions added using target_compile_definitions
  # were not added to the nvcc flags. This manually adds these definitions and
  # pass them to the options parameter in cuda_add_library
  string(APPEND cuda_cxx_flags " -DAF_WITH_NONFREE_SIFT")
endif()

include(kernel/scan_by_key/CMakeLists.txt)
include(kernel/thrust_sort_by_key/CMakeLists.txt)

cuda_add_library(afcuda
    scan.cu
    sort.hpp
    scan_by_key.cu
    kernel/convolve.cu
    kernel/convolve_separable.cu

    all.cu
    anisotropic_diffusion.cu
    any.cu
    approx.cu
    assign.cu
    bilateral.cu
    canny.cu
    cholesky.cu
    copy.cu
    count.cu
    diagonal.cu
    diff.cu
    dilate.cu
    dilate3d.cu
    erode.cu
    erode3d.cu
    exampleFunction.cu
    fast.cu
    fast_pyramid.cu
    fftconvolve.cu
    gradient.cu
    harris.cu
    histogram.cu
    homography.cu
    hsv_rgb.cu
    identity.cu
    iir.cu
    index.cu
    inverse.cu
    iota.cu
    ireduce.cu
    join.cu
    lookup.cu
    lu.cu
    match_template.cu
    max.cu
    mean.cu
    meanshift.cu
    medfilt.cu
    min.cu
    moments.cu
    nearest_neighbour.cu
    orb.cu
    pad_array_borders.cu
    product.cu
    qr.cu
    random_engine.cu
    range.cu
    regions.cu
    reorder.cu
    resize.cu
    rotate.cu
    select.cu
    set.cu
    sift.cu
    sobel.cu
    solve.cu
    sort.cu
    sort_by_key.cu
    sort_index.cu
    sparse.cu
    sparse_arith.cu
    sum.cu
    susan.cu
    svd.cu
    tile.cu
    topk.cu
    transform.cu
    transpose.cu
    transpose_inplace.cu
    triangle.cu
    unwrap.cu
    where.cu
    wrap.cu

    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/atomics.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/config.hpp
    kernel/convolve.hpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fast_lut.hpp
    kernel/fast_pyramid.hpp
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
    kernel/sift_nonfree.hpp
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
    Param.hpp
    anisotropic_diffusion.hpp
    approx.hpp
    arith.hpp
    assign.hpp
    backend.hpp
    bilateral.hpp
    binary.hpp
    blas.cpp
    blas.hpp
    canny.hpp
    cast.hpp
    cholesky.hpp
    complex.hpp
    convolve.cpp
    convolve.hpp
    copy.hpp
    cublas.cpp
    cublas.hpp
    cufft.cpp
    cufft.hpp
    cusolverDn.cpp
    cusolverDn.hpp
    cusparse.cpp
    cusparse.hpp
    debug_cuda.hpp
    diagonal.hpp
    diff.hpp
    driver.cpp
    err_cuda.hpp
    exampleFunction.hpp
    fast.hpp
    fast_pyramid.hpp
    fft.cpp
    fft.hpp
    fftconvolve.hpp
    GraphicsResourceManager.cpp
    GraphicsResourceManager.hpp
    gradient.hpp
    harris.hpp
    hist_graphics.cpp
    hist_graphics.hpp
    histogram.hpp
    homography.hpp
    hsv_rgb.hpp
    identity.hpp
    iir.hpp
    image.cpp
    image.hpp
    index.hpp
    inverse.hpp
    iota.hpp
    ireduce.hpp
    jit.cpp
    join.hpp
    logic.hpp
    lookup.hpp
    lu.hpp
    match_template.hpp
    math.cpp
    math.hpp
    mean.hpp
    meanshift.hpp
    medfilt.hpp
    memory.cpp
    memory.hpp
    moments.hpp
    morph.hpp
    morph3d_impl.hpp
    morph_impl.hpp
    nearest_neighbour.hpp
    orb.hpp
    platform.cpp
    platform.hpp
    plot.cpp
    plot.hpp
    print.hpp
    qr.hpp
    random_engine.hpp
    range.hpp
    reduce.hpp
    reduce_impl.hpp
    regions.hpp
    reorder.hpp
    resize.hpp
    rotate.hpp
    scalar.hpp
    scan.hpp
    scan_by_key.hpp
    select.hpp
    set.hpp
    shift.cpp
    shift.hpp
    sift.hpp
    sobel.hpp
    solve.hpp
    sort_by_key.hpp
    sort_index.hpp
    sparse.hpp
    sparse_arith.hpp
    sparse_blas.cpp
    sparse_blas.hpp
    surface.cpp
    surface.hpp
    susan.hpp
    svd.hpp
    tile.hpp
    topk.hpp
    traits.hpp
    transform.hpp
    transpose.hpp
    triangle.hpp
    types.hpp
    unary.hpp
    unwrap.hpp
    utility.hpp
    vector_field.cpp
    vector_field.hpp
    where.hpp
    wrap.hpp

    jit/BufferNode.hpp
    jit/kernel_generators.hpp

    OPTIONS "${cuda_cxx_flags} -Xcudafe \"--diag_suppress=1427\""
  )

arrayfire_set_default_cxx_flags(afcuda)
target_compile_definitions(afcuda PRIVATE AF_CUDA)

add_library(ArrayFire::afcuda ALIAS afcuda)

if(AF_WITH_NONFREE)
  target_compile_definitions(afcuda PRIVATE AF_WITH_NONFREE_SIFT)
endif()

add_dependencies(afcuda ${jit_kernel_targets})

target_include_directories (afcuda
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${ArrayFire_SOURCE_DIR}/src/api/c
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/jit
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(afcuda
    PRIVATE
      OpenMP::OpenMP_CXX
    )
elseif(NOT APPLE)
  message(FATAL_ERROR "OpenMP is required to compile CUDA Backend")
endif()

set_target_properties(afcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(afcuda
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    cuda_scan_by_key
    cuda_thrust_sort_by_key
    ${CUDA_LIBRARIES}
    ${CUDA_nvrtc_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
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

macro(afcu_collect_libs libname)
  if (WIN32)
    install(FILES       "${dlib_path_prefix}/${PX}${libname}64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}${SX}"
            DESTINATION ${AF_INSTALL_BIN_DIR}
            COMPONENT   cuda_dependencies)
  elseif (APPLE)
    get_filename_component(outpath "${dlib_path_prefix}/${PX}${libname}.${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}${SX}" REALPATH)
    install(FILES       "${outpath}"
            DESTINATION ${AF_INSTALL_BIN_DIR}
            RENAME      "${PX}${libname}.${CUDA_VERSION}${SX}"
            COMPONENT   cuda_dependencies)
  else () #UNIX
    get_filename_component(outpath "${dlib_path_prefix}/${PX}${libname}${SX}" REALPATH)
    install(FILES       ${outpath}
            DESTINATION ${AF_INSTALL_LIB_DIR}
            RENAME      "${PX}${libname}${SX}.${CUDA_VERSION}"
            COMPONENT   cuda_dependencies)
  endif ()
endmacro()

if(AF_INSTALL_STANDALONE)
  afcu_collect_libs(cufft)
  afcu_collect_libs(cublas)
  afcu_collect_libs(cusolver)
  afcu_collect_libs(cusparse)

  if(WIN32 AND ${CUDA_VERSION_MAJOR} EQUAL 10 AND ${CUDA_VERSION_MINOR} EQUAL 0)
    install(FILES       "${dlib_path_prefix}/${PX}nvrtc64_100_0${SX}"
            DESTINATION ${AF_INSTALL_BIN_DIR}
            COMPONENT   cuda_dependencies)
  else()
    afcu_collect_libs(nvrtc)
  endif()

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


source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/*|${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/*)
source_group("generated files"  FILES ${ArrayFire_BINARY_DIR}/version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h
                                REGULAR_EXPRESSION ${CMAKE_CURRENT_BINARY_DIR}/${kernel_headers_dir}/*)
source_group("" FILES CMakeLists.txt)
