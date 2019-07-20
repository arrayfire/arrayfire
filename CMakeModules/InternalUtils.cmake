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

function(arrayfire_get_platform_definitions variable)
if(WIN32)
  set(${variable} -DOS_WIN -DWIN32_LEAN_AND_MEAN -DNOMINMAX PARENT_SCOPE)
elseif(APPLE)
  set(${variable} -DOS_MAC PARENT_SCOPE)
elseif(UNIX)
  set(${variable} -DOS_LNX PARENT_SCOPE)
endif()
endfunction()

function(arrayfire_get_cuda_cxx_flags cuda_flags)
  if(NOT MSVC)
    set(flags "-std=c++14 --expt-relaxed-constexpr -Xcompiler -fPIC -Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}hidden")
  else()
    set(flags "-Xcompiler /wd4251 -Xcompiler /wd4068 -Xcompiler /wd4275 -Xcompiler /bigobj -Xcompiler /EHsc")
    if(CMAKE_GENERATOR MATCHES "Ninja")
      set(flags "${flags} -Xcompiler /FS")
    endif()
  endif()

  if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.3.0" AND
      ${CUDA_VERSION_MAJOR} LESS 8)
    set(flags "${flags} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED")
  endif()
  set(${cuda_flags} "${flags}" PARENT_SCOPE)
endfunction()

include(CheckCXXCompilerFlag)

function(arrayfire_set_default_cxx_flags target)
  arrayfire_get_platform_definitions(defs)
  target_compile_definitions(${target} PRIVATE ${defs})

  if(MSVC)
    target_compile_options(${target}
      PRIVATE
        /wd4251 /wd4068 /wd4275 /bigobj /EHsc)

    if(CMAKE_GENERATOR MATCHES "Ninja")
      target_compile_options(${target}
        PRIVATE
          /FS)
    endif()
  else()
    check_cxx_compiler_flag(-Wno-ignored-attributes has_ignored_attributes_flag)

    # OpenCL targets need this flag to avoid ignored attribute warnings in the
    # OpenCL headers
    if(has_ignored_attributes_flag)
        target_compile_options(${target}
          PRIVATE -Wno-ignored-attributes)
    endif()
  endif()
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

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)

  # Set a default build type if none was specified
  if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "The type of the build")
  endif()

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")

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

  if(APPLE)
    set(CMAKE_INSTALL_RPATH "/opt/arrayfire/lib")
  endif()

  include(WriteCompilerDetectionHeader)
  write_compiler_detection_header(
          FILE ${ArrayFire_BINARY_DIR}/include/af/compilers.h
          PREFIX AF
          COMPILERS AppleClang Clang GNU Intel MSVC
          # NOTE: cxx_attribute_deprecated does not work well with C
          FEATURES cxx_rvalue_references cxx_noexcept cxx_variadic_templates cxx_alignas cxx_static_assert
          ALLOW_UNKNOWN_COMPILERS
          #[VERSION <version>]
          #[PROLOG <prolog>]
          #[EPILOG <epilog>]
          )
endmacro()

mark_as_advanced(
    pkgcfg_lib_PC_CBLAS_cblas
    pkgcfg_lib_PC_LAPACKE_lapacke
    pkgcfg_lib_PKG_FFTW_fftw3
    )
