# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license. The complete license
# agreement can be obtained at: http://arrayfire.com/licenses/BSD-3-Clause
#
# A FindMKL script based on the recommendations by the Intel's Link Line
# Advisor. It currently only tested on the 2018 version of MKL on Windows,
# Linux, and OSX but it should work on older versions.
#
# To use this module call the mklvars.(sh,bat) script before you call cmake. This
# script is located in the bin folder of your mkl installation. This will set the
# MKLROOT environment variable which will be used to find the libraries on your system.
#
# In case you have oneAPI base toolkit installed, having ONEAPI_ROOT environment variable available
# also will enable picking Intel oneMKL automatically.
#
# Example:
# set(MKL_THREAD_LAYER "TBB")
# find_package(MKL)
#
# add_executable(myapp main.cpp)
# target_link_libraries(myapp PRIVATE MKL::Shared)
#
# This module bases its behavior based on the following variables:
#
# ``MKL_THREAD_LAYER``
#   The threading layer that needs to be used by the MKL library. This
#   Defines which library will be used to parallelize the MKL kernels. Possible
#   options are TBB(Default), GNU OpenMP, Intel OpenMP, Sequential
#
# This module provides the following :prop_tgt:'IMPORTED' targets:
#
# ``MKL::Shared``
#   Target used to define and link all MKL libraries required by Intel's Link
#   Line Advisor. This usually the only thing you need to link against unless
#   you want to link against the single dynamic library version of MKL
#   (libmkl_rt.so)
#
# ``MKL::Static``
#   Target used to define and link all MKL libraries required by Intel's Link
#   Line Advisor for a static build. This will still link the threading libraries
#   using dynamic linking as advised by the Intel Link Advisor
#
#  Optional:
#
# ``MKL::ThreadLayer{_STATIC}``
#   Target used to define the threading layer(TBB, OpenMP, etc.) based on
#   MKL_THREAD_LAYER variable.
#
# ``MKL::ThreadingLibrary``
#   Target used to define the threading library(libtbb, libomp, etc) that the
#   application will need to link against.
#
# ``MKL::Interface``
#   Target used to determine which interface library to use(32bit int or 64bit
#   int).
#
# ``MKL::Core``
#   Target for the dynamic library dispatcher
#
# ``MKL::RT``
#   Target for the single dynamic library
#
# ``MKL::{mkl_def;mkl_mc;mkl_mc3;mkl_avx;mkl_avx2;mkl_avx512}{_STATIC}``
#   Targets for MKL kernel libraries.
#
# This module has the following result variables:
#
# ``MKL_INTERFACE_INTEGER_SIZE``
#   This variable is set integer size in bytes on the platform where this module
#   runs. This is usually 4/8, and set of values this is dependent on MKL library.

include(CheckTypeSize)
include(FindPackageHandleStandardArgs)

check_type_size("int" INT_SIZE
  BUILTIN_TYPES_ONLY LANGUAGE C)

set(MKL_THREAD_LAYER "TBB" CACHE STRING "The thread layer to choose for MKL")
set_property(CACHE MKL_THREAD_LAYER PROPERTY STRINGS "TBB" "GNU OpenMP" "Intel OpenMP" "Sequential")

message(STATUS "MKL: Thread Layer(${MKL_THREAD_LAYER}) Interface(${INT_SIZE}-byte Integer)")

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
    $ENV{MKLROOT}
    $ENV{ONEAPI_ROOT}/mkl/latest
    /opt/intel/compilers_and_libraries/linux/mkl
  PATH_SUFFIXES
    include
    IntelSWTools/compilers_and_libraries/windows/mkl/include
    )

if(MKL_INCLUDE_DIR)
  mark_as_advanced(MKL_INCLUDE_DIR)
endif()

function(find_version)
  set(options "")
  set(single_args VAR FILE REGEX)
  set(multi_args "")
  cmake_parse_arguments(find_version "${options}" "${single_args}" "${multi_args}" ${ARGN})

  file(READ ${find_version_FILE} VERSION_FILE_CONTENTS)
  string(REGEX MATCH ${find_version_REGEX}
    VERSION_LINE "${VERSION_FILE_CONTENTS}")
  set(${ARGV0} ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()

if(MKL_INCLUDE_DIR)
  find_file(MKL_VERSION_HEADER
    NAMES
      mkl_version.h
    PATHS
      ${MKL_INCLUDE_DIR})

    find_version(MKL_MAJOR_VERSION
      FILE ${MKL_VERSION_HEADER}
      REGEX "__INTEL_MKL__ * ([0-9]+)")

    find_version(MKL_MINOR_VERSION
      FILE ${MKL_VERSION_HEADER}
      REGEX "__INTEL_MKL_MINOR__ * ([0-9]+)")

    find_version(MKL_UPDATE_VERSION
      FILE ${MKL_VERSION_HEADER}
      REGEX "__INTEL_MKL_UPDATE__ * ([0-9]+)")

    find_version(MKL_VERSION_MACRO
      FILE ${MKL_VERSION_HEADER}
      REGEX "INTEL_MKL_VERSION * ([0-9]+)")

  set(MKL_VERSION_STRING ${MKL_MAJOR_VERSION}.${MKL_MINOR_VERSION}.${MKL_UPDATE_VERSION})
  mark_as_advanced(MKL_VERSION_HEADER)
endif()


find_path(MKL_FFTW_INCLUDE_DIR
  NAMES
    fftw3_mkl.h
  HINTS
    ${MKL_INCLUDE_DIR}/fftw)
if(MKL_FFTW_INCLUDE_DIR)
  mark_as_advanced(MKL_FFTW_INCLUDE_DIR)
endif()

if(WIN32)
  if(${MSVC_VERSION} GREATER_EQUAL 1900)
    set(msvc_dir "vc_mt")
    set(shared_suffix "_dll")
    set(md_suffix "md")
  else()
    message(WARNING "MKL: MS Version not supported for MKL")
  endif()
endif()


if(WIN32)
  set(ENV_LIBRARY_PATHS "$ENV{LIB}")
  if (${CMAKE_VERSION} VERSION_GREATER 3.14)
    message(VERBOSE "MKL environment variable(LIB): ${ENV_LIBRARY_PATHS}")
  endif()
else()
  string(REGEX REPLACE ":" ";" ENV_LIBRARY_PATHS "$ENV{LIBRARY_PATH}")
  if (${CMAKE_VERSION} VERSION_GREATER 3.14)
    message(VERBOSE "MKL environment variable(LIBRARY_PATH): ${ENV_LIBRARY_PATHS}")
  endif()
endif()

# Finds and creates libraries for MKL with the MKL:: prefix
#
# Parameters:
#    NAME:         A variable name describing the library
#    LIBRARY_NAME: The library that needs to be searched
#
# OPTIONS:
#    DLL_ONLY     On Windows do not search for .lib files. Ignored in other
#                 platforms
#    SEARCH_STATIC Search for static versions of the libraries as well as the
#                  dynamic libraries
#
# Output Libraries:
#    MKL::${NAME}
#    MKL::${NAME}_STATIC
#
# Output Variables
#    MKL_${NAME}_LINK_LIBRARY:        on Unix: *.so on Windows *.lib
#    MKL_${NAME}_STATIC_LINK_LIBRARY: on Unix: *.a  on Windows *.lib
#    MKL_${NAME}_DLL_LIBRARY:         on Unix: ""   on Windows *.dll
function(find_mkl_library)
  set(options "SEARCH_STATIC;DLL_ONLY")
  set(single_args NAME LIBRARY_NAME)
  set(multi_args "")

  cmake_parse_arguments(mkl_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  if(TARGET MKL::${mkl_args_NAME})
    return()
  endif()

  add_library(MKL::${mkl_args_NAME}        SHARED IMPORTED)
  add_library(MKL::${mkl_args_NAME}_STATIC STATIC IMPORTED)

  if(NOT (WIN32 AND mkl_args_DLL_ONLY))
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.1;.so.2;.so.3;.so.4;.so.12")
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
        $ENV{MKLROOT}/lib
        $ENV{ONEAPI_ROOT}/mkl/latest/lib
        ${ENV_LIBRARY_PATHS}
        /opt/intel/compilers_and_libraries/linux/mkl/lib
      PATH_SUFFIXES
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
        ""
        intel64
        intel64/gcc4.7)
    list(REMOVE_ITEM CMAKE_FIND_LIBRARY_SUFFIXES ".so.1")
    if(MKL_${mkl_args_NAME}_LINK_LIBRARY)
      if (CMAKE_VERSION VERSION_GREATER 3.14)
        message(VERBOSE "MKL_${mkl_args_NAME}_LINK_LIBRARY: ${MKL_${mkl_args_NAME}_LINK_LIBRARY}")
      endif()
      mark_as_advanced(MKL_${mkl_args_NAME}_LINK_LIBRARY)
    endif()
  endif()

  #message(STATUS "NAME: ${mkl_args_NAME} LIBNAME: ${mkl_args_LIBRARY_NAME} MKL_${mkl_args_NAME}_LINK_LIBRARY  ${MKL_${mkl_args_NAME}_LINK_LIBRARY}")

  if(mkl_args_SEARCH_STATIC)
    find_library(MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY
      NAMES
        ${CMAKE_STATIC_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
      PATHS
        /opt/intel/mkl/lib
        /opt/intel/tbb/lib
        /opt/intel/lib
        $ENV{MKLROOT}/lib
        $ENV{ONEAPI_ROOT}/mkl/latest/lib
        ${ENV_LIBRARY_PATHS}
        /opt/intel/compilers_and_libraries/linux/mkl/lib
      PATH_SUFFIXES
        ""
        intel64
        intel64/gcc4.7
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
        IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
        )
    if(MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY)
      if(CMAKE_VERSION VERSION_GREATER 3.14)
        message(VERBOSE "MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY: ${MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY}")
      endif()
    endif()
    mark_as_advanced(MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY)
  endif()

  set_target_properties(MKL::${mkl_args_NAME}
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
      IMPORTED_LOCATION "${MKL_${mkl_args_NAME}_LINK_LIBRARY}"
      IMPORTED_NO_SONAME TRUE)

  set_target_properties(MKL::${mkl_args_NAME}_STATIC
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
      IMPORTED_LOCATION "${MKL_${mkl_args_NAME}_STATIC_LINK_LIBRARY}"
      IMPORTED_NO_SONAME TRUE)

  if(WIN32)
    find_file(MKL_${mkl_args_NAME}_DLL_LIBRARY
      NAMES
        ${CMAKE_SHARED_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_SHARED_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${md_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
        lib${mkl_args_LIBRARY_NAME}${md_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
        $ENV{LIB}
        $ENV{LIBRARY_PATH}
      PATH_SUFFIXES
        IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl
        IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler
        IntelSWTools/compilers_and_libraries/windows/redist/intel64/tbb/${msvc_dir}
      NO_SYSTEM_ENVIRONMENT_PATH)

    set_target_properties(MKL::${mkl_args_NAME}
      PROPERTIES
        IMPORTED_LOCATION "${MKL_${mkl_args_NAME}_DLL_LIBRARY}"
        IMPORTED_IMPLIB "${MKL_${mkl_args_NAME}_LINK_LIBRARY}")

    mark_as_advanced(MKL_${mkl_args_NAME}_DLL_LIBRARY)
  endif()
endfunction()


find_mkl_library(NAME Core LIBRARY_NAME mkl_core SEARCH_STATIC)
find_mkl_library(NAME RT LIBRARY_NAME mkl_rt)

if(AF_BUILD_ONEAPI)
    find_mkl_library(NAME Sycl LIBRARY_NAME sycl DLL_ONLY)
	find_mkl_library(NAME SyclLapack LIBRARY_NAME sycl_lapack DLL_ONLY)
	find_mkl_library(NAME SyclDft LIBRARY_NAME sycl_dft DLL_ONLY)
	find_mkl_library(NAME SyclBlas LIBRARY_NAME sycl_blas DLL_ONLY)
	find_mkl_library(NAME SyclSparse LIBRARY_NAME sycl_sparse DLL_ONLY)
endif()

# MKL can link against Intel OpenMP, GNU OpenMP, TBB, and Sequential
if(MKL_THREAD_LAYER STREQUAL "Intel OpenMP")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_intel_thread SEARCH_STATIC)
  find_mkl_library(NAME ThreadingLibrary LIBRARY_NAME iomp5)
elseif(MKL_THREAD_LAYER STREQUAL "GNU OpenMP")
  find_package(OpenMP REQUIRED)
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_gnu_thread SEARCH_STATIC)
  set(MKL_ThreadingLibrary_LINK_LIBRARY ${OpenMP_gomp_LIBRARY})
  if(MKL_ThreadingLibrary_LINK_LIBRARY)
    mark_as_advanced(MKL_${mkl_args_NAME}_LINK_LIBRARY)
  endif()
  if(NOT TARGET MKL::ThreadingLibrary)
    add_library(MKL::ThreadingLibrary SHARED IMPORTED)
    set_target_properties(MKL::ThreadingLibrary
      PROPERTIES
        IMPORTED_LOCATION "${MKL_ThreadingLibrary_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
  endif()
elseif(MKL_THREAD_LAYER STREQUAL "TBB")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_tbb_thread SEARCH_STATIC)
  find_mkl_library(NAME ThreadingLibrary LIBRARY_NAME tbb)
elseif(MKL_THREAD_LAYER STREQUAL "Sequential")
  find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_sequential SEARCH_STATIC)
endif()

if("${INT_SIZE}" EQUAL 4)
  set(MKL_INTERFACE_INTEGER_SIZE 4)
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_lp64 SEARCH_STATIC)
else()
  set(MKL_INTERFACE_INTEGER_SIZE 8)
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_ilp64 SEARCH_STATIC)
endif()

set(MKL_KernelLibraries "mkl_def;mkl_mc;mkl_mc3;mkl_avx;mkl_avx2;mkl_avx512")

foreach(lib ${MKL_KernelLibraries})
  find_mkl_library(NAME ${lib} LIBRARY_NAME ${lib} DLL_ONLY)

  if(MKL_${lib}_LINK_LIBRARY)
    list(APPEND MKL_RUNTIME_KERNEL_LIBRARIES_TMP ${MKL_${lib}_LINK_LIBRARY})
  endif()

  if(MKL_${lib}_DLL_LIBRARY)
    list(APPEND MKL_RUNTIME_KERNEL_LIBRARIES_TMP ${MKL_${lib}_DLL_LIBRARY})
  endif()
endforeach()

set(MKL_RUNTIME_KERNEL_LIBRARIES "${MKL_RUNTIME_KERNEL_LIBRARIES_TMP}" CACHE STRING
    "MKL kernel libraries targeting different CPU architectures")
mark_as_advanced(MKL_RUNTIME_KERNEL_LIBRARIES)

# Bypass developer warning that the first argument to find_package_handle_standard_args (MKL_...) does not match
# the name of the calling package (MKL)
# https://cmake.org/cmake/help/v3.17/module/FindPackageHandleStandardArgs.html
set(FPHSA_NAME_MISMATCHED TRUE)

find_package_handle_standard_args(MKL_Shared
  FAIL_MESSAGE "Could NOT find MKL: Source the compilervars.sh or mklvars.sh scripts included with your installation of MKL. This script searches for the libraries in MKLROOT, LIBRARY_PATHS(Linux), and LIB(Windows) environment variables"
  VERSION_VAR  MKL_VERSION_STRING
  REQUIRED_VARS MKL_INCLUDE_DIR
                MKL_Core_LINK_LIBRARY
                MKL_Interface_LINK_LIBRARY
                MKL_ThreadLayer_LINK_LIBRARY)

find_package_handle_standard_args(MKL_Static
  FAIL_MESSAGE "Could NOT find MKL: Source the compilervars.sh or mklvars.sh scripts included with your installation of MKL. This script searches for the libraries in MKLROOT, LIBRARY_PATHS(Linux), and LIB(Windows) environment variables"
  VERSION_VAR   MKL_VERSION_STRING
  REQUIRED_VARS MKL_INCLUDE_DIR
                MKL_Core_STATIC_LINK_LIBRARY
                MKL_Interface_STATIC_LINK_LIBRARY
                MKL_ThreadLayer_STATIC_LINK_LIBRARY)

if(NOT WIN32)
  find_library(M_LIB m)
  mark_as_advanced(M_LIB)
endif()

if(TARGET MKL::RT)
  set_target_properties(MKL::RT
  PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}")
endif()

if(MKL_Shared_FOUND AND NOT TARGET MKL::Shared)
  add_library(MKL::Shared SHARED IMPORTED)
  if(MKL_THREAD_LAYER STREQUAL "Sequential")
    set_target_properties(MKL::Shared
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::Interface;MKL::ThreadLayer;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
  else()
    set_target_properties(MKL::Shared
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::Interface;MKL::ThreadLayer;MKL::ThreadingLibrary;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
  endif()
  if(WIN32)
    set_target_properties(MKL::Shared
      PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_DLL_LIBRARY}"
        IMPORTED_IMPLIB "${MKL_Core_LINK_LIBRARY}")
  endif()
endif()

if(MKL_Static_FOUND AND NOT TARGET MKL::Static)
  add_library(MKL::Static STATIC IMPORTED)

  if(UNIX AND NOT APPLE)
    if(MKL_THREAD_LAYER STREQUAL "Sequential")
      set_target_properties(MKL::Static
        PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_STATIC_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "-Wl,--start-group;MKL::Core_STATIC;MKL::Interface_STATIC;MKL::ThreadLayer_STATIC;-Wl,--end-group;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
    else()
      set_target_properties(MKL::Static
        PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_STATIC_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "-Wl,--start-group;MKL::Core_STATIC;MKL::Interface_STATIC;MKL::ThreadLayer_STATIC;-Wl,--end-group;MKL::ThreadingLibrary;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
    endif()
  else()
    if(MKL_THREAD_LAYER STREQUAL "Sequential")
      set_target_properties(MKL::Static
        PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_STATIC_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::Core_STATIC;MKL::Interface_STATIC;MKL::ThreadLayer_STATIC;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
    else()
      set_target_properties(MKL::Static
        PROPERTIES
        IMPORTED_LOCATION "${MKL_Core_STATIC_LINK_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "MKL::Core_STATIC;MKL::Interface_STATIC;MKL::ThreadLayer_STATIC;MKL::ThreadingLibrary;${CMAKE_DL_LIBS};${M_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}"
        IMPORTED_NO_SONAME TRUE)
    endif()
  endif()
endif()

set(MKL_FOUND OFF)
if(MKL_Shared_FOUND OR MKL_Static_FOUND)
  set(MKL_FOUND ON)
endif()
