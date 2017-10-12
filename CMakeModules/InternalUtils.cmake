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

macro(arrayfire_set_cmake_default_variables)
  set(CMAKE_PREFIX_PATH "${ArrayFire_BINARY_DIR}/cmake;${CMAKE_PREFIX_PATH}")
  set(BUILD_SHARED_LIBS ON)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_EXTENSIONS OFF)

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
    message(WARNING "Code Coverage in Visual Studio is not tested")
    set(CMAKE_CXX_FLAGS_COVERAGE "")
    set(CMAKE_C_FLAGS_COVERAGE "")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "/OPT:NOREF /PROFILE")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "/OPT:NOREF /PROFILE")
    set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE "/OPT:NOREF /PROFILE")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "/OPT:NOREF /PROFILE")
  endif()

  mark_as_advanced(
      CMAKE_CXX_FLAGS_COVERAGE
      CMAKE_C_FLAGS_COVERAGE
      CMAKE_EXE_LINKER_FLAGS_COVERAGE
      CMAKE_SHARED_LINKER_FLAGS_COVERAGE
      CMAKE_STATIC_LINKER_FLAGS_COVERAGE )

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Store all binaries in the bin/<Config> directory
  if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ArrayFire_BINARY_DIR}/bin)
  endif()

  if(APPLE)
    # TODO(umar) Remove rpath to third_party lib
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR};${ArrayFire_BINARY_DIR}/third_party/forge/lib")
  endif()
endmacro()
