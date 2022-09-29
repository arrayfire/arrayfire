# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

function(dependency_check VAR ERROR_MESSAGE)
  if(NOT ${VAR})
    message(SEND_ERROR ${ERROR_MESSAGE})
  endif()
endfunction()

# Includes the directory if the variable is set
function(conditional_directory variable directory)
  if(${variable})
    add_subdirectory(${directory})
  endif()
endfunction()

include(CheckCXXCompilerFlag)

if(WIN32)
  check_cxx_compiler_flag(/Zc:__cplusplus cplusplus_define)
  check_cxx_compiler_flag(/permissive- cxx_compliance)
endif()

function(arrayfire_set_default_cxx_flags target)
  target_compile_options(${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
              # C4068: Warnings about unknown pragmas
              # C4668: Warnings about unknown defintions
              # C4275: Warnings about using non-exported classes as base class of an
              #        exported class
              $<$<CXX_COMPILER_ID:MSVC>:  /wd4251
                                          /wd4068
                                          /wd4275
                                          /wd4668
                                          /wd4710
                                          /wd4505
                                          /bigobj
                                          /EHsc
                                          # MSVC incorrectly sets the cplusplus to 199711L even if the compiler supports
                                          # c++11 features. This flag sets it to the correct standard supported by the
                                          # compiler
                                          $<$<BOOL:${cplusplus_define}>:/Zc:__cplusplus>
                                          $<$<BOOL:${cxx_compliance}>:/permissive-> >

              # OpenCL targets need this flag to avoid
              # ignored attribute warnings in the OpenCL
              # headers
              $<$<BOOL:${has_ignored_attributes_flag}>:-Wno-ignored-attributes>
              $<$<BOOL:${has_all_warnings_flag}>:-Wall>>
    )

  target_compile_definitions(${target}
    PRIVATE
      AFDLL
      $<$<PLATFORM_ID:Windows>:             OS_WIN
                                            WIN32_LEAN_AND_MEAN
                                            NOMINMAX>
      $<$<PLATFORM_ID:Darwin>:              OS_MAC>
      $<$<PLATFORM_ID:Linux>:               OS_LNX>

      $<$<BOOL:${AF_WITH_LOGGING}>:           AF_WITH_LOGGING>
      $<$<BOOL:${AF_CACHE_KERNELS_TO_DISK}>:  AF_CACHE_KERNELS_TO_DISK>
  )
endfunction()

function(__af_deprecate_var var access value)
  if(access STREQUAL "READ_ACCESS")
    message(DEPRECATION "Variable ${var} is deprecated. Use AF_${var} instead.")
  endif()
endfunction()

function(af_deprecate var newvar)
  if(DEFINED ${var})
    message(DEPRECATION "Variable ${var} is deprecated. Use ${newvar} instead.")
    get_property(doc CACHE ${newvar} PROPERTY HELPSTRING)
    set(${newvar} ${${var}} CACHE BOOL "${doc}" FORCE)
    unset(${var} CACHE)
  endif()
  variable_watch(${var} __af_deprecate_var)
endfunction()

function(get_native_path out_path path)
  file(TO_NATIVE_PATH ${path} native_path)
  if (WIN32)
    string(REPLACE "\\" "\\\\" native_path  ${native_path})
    set(${out_path} ${native_path} PARENT_SCOPE)
  else ()
    set(${out_path} ${path} PARENT_SCOPE)
  endif ()
endfunction()

macro(arrayfire_set_cmake_default_variables)
  set(CMAKE_PREFIX_PATH "${ArrayFire_BINARY_DIR};${CMAKE_PREFIX_PATH}")
  set(BUILD_SHARED_LIBS ON)

  set(CMAKE_CXX_FLAGS_COVERAGE
      "-g -O0"
      CACHE STRING "Flags used by the C++ compiler during coverage builds.")

  set(CMAKE_C_FLAGS_COVERAGE
      "-g -O0"
      CACHE STRING "Flags used by the C compiler during coverage builds.")
  set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
      ""
      CACHE STRING "Flags used for linking binaries during coverage builds.")
  set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
      ""
      CACHE STRING "Flags used by the shared libraries linker during coverage builds.")
  set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE
      ""
      CACHE STRING "Flags used by the static libraries linker during coverage builds.")
  set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE
      ""
      CACHE STRING "Flags used by the module linker during coverage builds.")

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE} --coverage")
    set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_COVERAGE} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_COVERAGE} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_COVERAGE} --coverage")
    set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE "${CMAKE_STATIC_LINKER_FLAGS_COVERAGE}")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "${CMAKE_STATIC_LINKER_FLAGS_COVERAGE} --coverage")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_COVERAGE "")
    set(CMAKE_C_FLAGS_COVERAGE "")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "")
    set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE "")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "")
  endif()

  mark_as_advanced(
      CMAKE_CXX_FLAGS_COVERAGE
      CMAKE_C_FLAGS_COVERAGE
      CMAKE_EXE_LINKER_FLAGS_COVERAGE
      CMAKE_SHARED_LINKER_FLAGS_COVERAGE
      CMAKE_STATIC_LINKER_FLAGS_COVERAGE
      CMAKE_MODULE_LINKER_FLAGS_COVERAGE)

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Store all binaries in the bin/<Config> directory
  if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ArrayFire_BINARY_DIR}/bin)
  endif()

  if(APPLE AND (NOT DEFINED CMAKE_INSTALL_RPATH))
      message(WARNING "CMAKE_INSTALL_RPATH is required when installing ArrayFire to the local system. Set it to /opt/arrayfire/lib if making the installer or your own custom install path.")
  endif()

  # This code is used to generate the compilers.h file in CMakeModules. Not all
  # features of this modules are supported in the versions of CMake we wish to
  # support so we are directly including the files here
  #  set(compiler_header_epilogue [=[
  #  #if defined(AF_COMPILER_CXX_RELAXED_CONSTEXPR) && AF_COMPILER_CXX_RELAXED_CONSTEXPR
  #  #define AF_CONSTEXPR constexpr
  #  #else
  #  #define AF_CONSTEXPR
  #  #endif
  #  ]=])
  #  include(WriteCompilerDetectionHeader)
  #  write_compiler_detection_header(
  #          FILE ${ArrayFire_BINARY_DIR}/include/af/compilers.h
  #          PREFIX AF
  #          COMPILERS AppleClang Clang GNU Intel MSVC
  #          # NOTE: cxx_attribute_deprecated does not work well with C
  #          FEATURES cxx_rvalue_references cxx_noexcept cxx_variadic_templates cxx_alignas
  #          cxx_static_assert cxx_generalized_initializers cxx_relaxed_constexpr
  #          ALLOW_UNKNOWN_COMPILERS
  #          #[VERSION <version>]
  #          #[PROLOG <prolog>]
  #          EPILOG ${compiler_header_epilogue}
  #          )
  configure_file(
    ${ArrayFire_SOURCE_DIR}/CMakeModules/compilers.h
    ${ArrayFire_BINARY_DIR}/include/af/compilers.h)
endmacro()

macro(set_policies)
  cmake_parse_arguments(SP "" "TYPE" "POLICIES" ${ARGN})
  foreach(_policy ${SP_POLICIES})
    if(POLICY ${_policy})
      cmake_policy(SET ${_policy} ${SP_TYPE})
    endif()
  endforeach()
endmacro()

macro(af_mkl_batch_check)
  set(CMAKE_REQUIRED_LIBRARIES "MKL::RT")
  check_symbol_exists(sgetrf_batch_strided "mkl_lapack.h" MKL_BATCH)
endmacro()

# Creates a CACHEd CMake variable which has limited set of possible string values
# Argumehts:
#   NAME: The name of the variable
#   DEFAULT: The default value of the variable
#   DESCRIPTION: The description of the variable
#   OPTIONS: The possible set of values for the option
#
# Example:
#
# af_multiple_option(NAME        AF_COMPUTE_LIBRARY
#                    DEFAULT     "Intel-MKL"
#                    DESCRIPTION "Compute library for signal processing and linear algebra routines"
#                    OPTIONS     "Intel-MKL" "FFTW/LAPACK/BLAS")
macro(af_multiple_option)
  cmake_parse_arguments(opt "" "NAME;DEFAULT;DESCRIPTION" "OPTIONS" ${ARGN})
  set(${opt_NAME} ${opt_DEFAULT} CACHE STRING ${opt_DESCRIPTION})
  set_property(CACHE ${opt_NAME} PROPERTY STRINGS ${opt_OPTIONS})
endmacro()

mark_as_advanced(
    pkgcfg_lib_PC_CBLAS_cblas
    pkgcfg_lib_PC_LAPACKE_lapacke
    pkgcfg_lib_PKG_FFTW_fftw3
    )
