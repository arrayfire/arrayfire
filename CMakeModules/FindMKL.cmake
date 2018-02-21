# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
#
# A FindMKL script based on the recommendations by the Intel's Link Line
# Advisor. It currently only tested on the 2018 version of MKL on Windows,
# Linux, and OSX but it should work on older versions. It creates an MKL::MKL
# library which has the required dependencies to for a dynamic link based
# on the advisor's output.
include(CheckTypeSize)

check_type_size("int" INT_SIZE
  BUILTIN_TYPES_ONLY LANGUAGE C)

set(MKL_THREAD_LAYER "TBB" CACHE STRING "The thread layer to choose for MKL")
set_property(CACHE MKL_THREAD_LAYER PROPERTY STRINGS "TBB" "GNU OpenMP" "Intel OpenMP" "Sequential")

if(NOT MKL_THREAD_LAYER STREQUAL MKL_THREAD_LAYER_LAST)
  unset(MKL::ThreadLayer CACHE)
  unset(MKL::ThreadingLibrary CACHE)
  unset(MKL_ThreadLayer_LINK_LIBRARY CACHE)
  unset(MKL_ThreadLayer_STATIC_LINK_LIBRARY CACHE)
  unset(MKL_ThreadLayer_DLL_LIBRARY CACHE)
  unset(MKL_ThreadingLibrary_LINK_LIBRARY CACHE)
  unset(MKL_ThreadingLibrary_STATIC_LINK_LIBRARY CACHE)
  unset(MKL_ThreadingLibrary_DLL_LIBRARY CACHE)
  set(MKL_THREAD_LAYER_LAST ${MKL_THREAD_LAYER} CACHE INTERNAL "" FORCE)
endif()

find_path(MKL_INCLUDE_DIR
  NAMES
    mkl.h
    mkl_blas.h
    mkl_cblas.h
  PATHS
    /opt/intel
    /opt/intel/mkl
    $ENV{MKL_ROOT}
  PATH_SUFFIXES
    include
    IntelSWTools/compilers_and_libraries/windows/mkl/include
    )

find_path(MKL_FFTW_INCLUDE_DIR
  NAMES
    fftw3_mkl.h
  HINTS
    ${MKL_INCLUDE_DIR}/fftw)

if(WIN32)
  if(${MSVC_VERSION} GREATER_EQUAL 1900)
    set(msvc_dir "vc14")
    set(shared_suffix "_dll")
    set(md_suffix "md")
  else()
    message(WARNING "MKL: MS Version not supported for MKL")
  endif()
endif()

# Finds and creates libraries for MKL with the MKL:: prefix
#
# Parameters:
#    NAME:         A variable name describing the library
#    LIBRARY_NAME: The library that needs to be searched
#
# Output Libraries:
#    MKL::${NAME}
#    MKL::${NAME}_STATIC
#
# Output Variables
#    MKL_INCLUDE_DIR:                Include directory for MKL
#    MKL_FFTW_INCLUDE_DIR:           Include directory for the MKL FFTW interface
#    MKL_${NAME}_LINK_LIBRARY:        on Unix: *.so on Windows *.lib
#    MKL_${NAME}_STATIC_LINK_LIBRARY: on Unix: *.a  on Windows *.lib
#    MKL_${NAME}_DLL_LIBRARY:         on Unix: ""   on Windows *.dll
function(find_mkl_library)
  set(options "")
  set(single_args NAME LIBRARY_NAME)
  set(multi_args "")

  cmake_parse_arguments(mkl_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  add_library(MKL::${mkl_args_NAME}        SHARED IMPORTED)
  add_library(MKL::${mkl_args_NAME}_STATIC SHARED IMPORTED)
  find_library(MKL_${mkl_args_NAME}_LINK_LIBRARY
    NAMES
      ${mkl_args_LIBRARY_NAME}${shared_suffix}
      ${mkl_args_LIBRARY_NAME}${md_suffix}
      lib${mkl_args_LIBRARY_NAME}${md_suffix}
      ${mkl_args_LIBRARY_NAME}
    PATHS
      /opt/intel/mkl/lib
      /opt/intel/tbb/lib
      /opt/intel/lib
      $ENV{MKL_ROOT}/lib
    PATH_SUFFIXES
      IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
      ""
      intel64
      intel64/gcc4.7)
  mark_as_advanced(MKL_${mkl_args_NAME}_LINK_LIBRARY)

  #message(STATUS "NAME: ${mkl_args_NAME} LIBNAME: ${mkl_args_LIBRARY_NAME} MKL_${mkl_args_NAME}_LINK_LIBRARY  ${MKL_${mkl_args_NAME}_LINK_LIBRARY}")

  # The rt library does not have a static library
  if(NOT ${mkl_args_NAME} STREQUAL "rt")
    find_library(MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY
      NAMES
        ${CMAKE_STATIC_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
      PATHS
        /opt/intel/mkl/lib
        /opt/intel/tbb/lib
        /opt/intel/lib
        $ENV{MKL_ROOT}/lib
      PATH_SUFFIXES
        ""
        intel64
        intel64/gcc4.7
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
        )
      mark_as_advanced(MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY)
    endif()

    set_target_properties(MKL::${mkl_args_NAME}
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        IMPORTED_LOCATION "${MKL_${mkl_args_NAME}_LINK_LIBRARY}"
        IMPORTED_NO_SONAME TRUE)
    if(WIN32)
      find_file(MKL_${mkl_args_NAME}_DLL_LIBRARY
        NAMES
          ${CMAKE_SHARED_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${CMAKE_SHARED_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${md_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
          lib${mkl_args_LIBRARY_NAME}${md_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
        PATH_SUFFIXES
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/tbb/${msvc_dir}
        NO_SYSTEM_ENVIRONMENT_PATH)

      set_target_properties(MKL::${mkl_args_NAME}
        PROPERTIES
          IMPORTED_LOCATION "${MKL_${mkl_args_NAME}_DLL_LIBRARY}"
          IMPORTED_IMPLIB "${MKL_${mkl_args_NAME}_LINK_LIBRARY}")
    endif()
endfunction()


find_mkl_library(NAME Core LIBRARY_NAME mkl_core)
find_mkl_library(NAME RT LIBRARY_NAME mkl_rt)

# MKL can link against Intel OpenMP, GNU OpenMP, TBB, and Sequential
if(MKL_THREAD_LAYER STREQUAL "Intel OpenMP")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_intel_thread)
  find_mkl_library(NAME ThreadingLibrary LIBRARY_NAME iomp5)
elseif(MKL_THREAD_LAYER STREQUAL "GNU OpenMP")
  find_package(OpenMP REQUIRED)
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_gnu_thread)
  set(MKL::ThreadingLibrary OpenMP::OpenMP_CXX CACHE STRING "The OpenMP Threading Library")
elseif(MKL_THREAD_LAYER STREQUAL "TBB")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_tbb_thread)
  find_mkl_library(NAME ThreadingLibrary LIBRARY_NAME tbb)
elseif(MKL_THREAD_LAYER STREQUAL "Sequential")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_sequential)
endif()

if("${INT_SIZE}" EQUAL 4)
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_lp64)
else()
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_ilp64)
endif()

set(MKL_RUNTIME_KERNEL_LIBRARIES "" CACHE FILEPATH "MKL kernel libraries targeting different CPU architectures")
set(MKL_KernelLibraries "mkl_def;mkl_mc;mkl_mc3;mkl_avx;mkl_avx2;mkl_avx512")

foreach(lib ${MKL_KernelLibraries})
  find_mkl_library(NAME ${lib} LIBRARY_NAME ${lib})
  if(MKL_${lib}_LINK_LIBRARY OR MKL_${lib}_DLL_LIBRARY)
    list(APPEND MKL_RUNTIME_KERNEL_LIBRARIES $<TARGET_FILE:MKL::${lib}>)
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
  REQUIRED_VARS MKL_INCLUDE_DIR MKL_Core_LINK_LIBRARY)
if(NOT WIN32)
  find_library(M_LIB m)
endif()
if(MKL_FOUND)
  add_library(MKL::MKL SHARED IMPORTED)
  if(MKL_THREAD_LAYER STREQUAL "Sequential")
    set_target_properties(MKL::MKL
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::ThreadLayer;MKL::Interface;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
  else()
    set_target_properties(MKL::MKL
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::ThreadLayer;MKL::Interface;MKL::ThreadingLibrary;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
  endif()
  if(WIN32)
    set_target_properties(MKL::MKL
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_DLL_LIBRARY}"
        IMPORTED_IMPLIB "${MKL_Core_LINK_LIBRARY}")
  endif()
endif()
