# Fetched the original content of this file from
# https://github.com/soumith/cudnn.torch
#
# Original Copyright:
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
#
# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
#
# FindcuDNN
# -------
#
# Find cuDNN library
#
# This module creates imported target cuDNN::cuDNN upon successfull
# lookup of cuDNN headers and libraries.
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIET>: as usual
#
# Usage
# -----
# add_exectuable(helloworld main.cpp)
# target_link_libraries(helloworld PRIVATE cuDNN::cuDNN)
#
# Note: It is recommended to avoid using variables set by the find module.
#
# Result variables
# ----------------
#
# This module will set the following variables in your project:
#
# ``cuDNN_INCLUDE_DIRS``
#   where to find cudnn.h.
# ``cuDNN_LINK_LIBRARY``
#   the libraries to link against to use cuDNN.
# ``cuDNN_DLL_LIBRARY``
#   Windows DLL of cuDNN
# ``cuDNN_FOUND``
#   If false, do not try to use cuDNN.
# ``cuDNN_VERSION``
#   Version of the cuDNN library we looked for

find_package(PkgConfig)
pkg_check_modules(PC_CUDNN QUIET cuDNN)

find_package(CUDA QUIET)

find_path(cuDNN_INCLUDE_DIRS
  NAMES cudnn.h
  HINTS
    ${cuDNN_ROOT_DIR}
    ${PC_CUDNN_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE}
  PATH_SUFFIXES include
  DOC "cuDNN include directory path." )

if(cuDNN_INCLUDE_DIRS)
  file(READ ${cuDNN_INCLUDE_DIRS}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
    CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  list(LENGTH CUDNN_MAJOR_VERSION cudnn_ver_matches)
  if(${cudnn_ver_matches} EQUAL 0)
    file(READ ${cuDNN_INCLUDE_DIRS}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
      CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  endif()
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
      CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
    CUDNN_MINOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
      CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
    CUDNN_PATCH_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
      CUDNN_PATCH_VERSION "${CUDNN_PATCH_VERSION}")
  set(cuDNN_VERSION ${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION})
endif()

# Choose lib suffix to be exact major version if requested
# otherwise, just pick the one read from cudnn.h header
if(cuDNN_FIND_VERSION_EXACT)
  set(cudnn_ver_suffix "${cuDNN_FIND_VERSION_MAJOR}")
else()
  set(cudnn_ver_suffix "${CUDNN_MAJOR_VERSION}")
endif()

if(cuDNN_INCLUDE_DIRS)
  get_filename_component(libpath_cudart "${CUDA_CUDART_LIBRARY}" PATH)

  find_library(cuDNN_LINK_LIBRARY
    NAMES
      libcudnn.so.${cudnn_ver_suffix}
      libcudnn.${cudnn_ver_suffix}.dylib
      cudnn
    PATHS
      ${cuDNN_ROOT_DIR}
      ${PC_CUDNN_LIBRARY_DIRS}
      $ENV{LD_LIBRARY_PATH}
      ${libpath_cudart}
      ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin lib/x64 bin/x64
    DOC "cuDNN link library." )

  if(WIN32 AND cuDNN_LINK_LIBRARY)
    find_file(cuDNN_DLL_LIBRARY
    NAMES cudnn64_${cudnn_ver_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
    PATHS
      ${cuDNN_ROOT_DIR}
      ${PC_CUDNN_LIBRARY_DIRS}
      $ENV{PATH}
      ${libpath_cudart}
      ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin lib/x64 bin/x64
    DOC "cuDNN Windows DLL." )
  endif()
endif()

find_package_handle_standard_args(cuDNN
  REQUIRED_VARS cuDNN_LINK_LIBRARY cuDNN_INCLUDE_DIRS
  VERSION_VAR   cuDNN_VERSION)

mark_as_advanced(cuDNN_LINK_LIBRARY cuDNN_INCLUDE_DIRS cuDNN_DLL_LIBRARY)

if(cuDNN_FOUND)
  add_library(cuDNN::cuDNN SHARED IMPORTED)
  if(WIN32)
    set_target_properties(cuDNN::cuDNN
      PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${cuDNN_DLL_LIBRARY}"
      IMPORTED_IMPLIB "${cuDNN_LINK_LIBRARY}"
    )
  else(WIN32)
    set_target_properties(cuDNN::cuDNN
      PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${cuDNN_LINK_LIBRARY}"
    )
  endif(WIN32)
endif(cuDNN_FOUND)
