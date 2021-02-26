# Fetched the original content of this file from
# https://github.com/soumith/cudnn.torch
#
# Original Copyright:
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
#
# Copyright (c) 2021, ArrayFire
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
#
# ``cuDNN_LINK_LIBRARY``
#   the libraries to link against to use cuDNN. Priot to cuDNN 8, this is a huge monolithic
#   library. However, since cuDNN 8 it has been split into multiple shared libraries. If
#   cuDNN version 8 if found, this variable contains the shared library that dlopens the
#   other libraries: cuDNN_*_INFER_LINK_LIBRARY and cuDNN_*_TRAIN_LINK_LIBRARY as needed.
#   For versions of cuDNN 7 or lower, cuDNN_*_INFER_LINK_LIBRARY and cuDNN_*_TRAIN_LINK_LIBRARY
#   are not defined.
#
# ``cuDNN_ADV_INFER_LINK_LIBRARY``
#   the libraries to link directly to use advanced inference API from cuDNN.
# ``cuDNN_ADV_INFER_DLL_LIBRARY``
#   Corresponding advanced inference API Windows DLL. This is not set on non-Windows platforms.
# ``cuDNN_ADV_TRAIN_LINK_LIBRARY``
#   the libraries to link directly to use advanced training API from cuDNN.
# ``cuDNN_ADV_TRAIN_DLL_LIBRARY``
#   Corresponding advanced training API Windows DLL. This is not set on non-Windows platforms.
#
# ``cuDNN_CNN_INFER_LINK_LIBRARY``
#   the libraries to link directly to use convolutional nueral networks inference API from cuDNN.
# ``cuDNN_CNN_INFER_DLL_LIBRARY``
#   Corresponding CNN inference API Windows DLL. This is not set on non-Windows platforms.
# ``cuDNN_CNN_TRAIN_LINK_LIBRARY``
#   the libraries to link directly to use convolutional nueral networks training API from cuDNN.
# ``cuDNN_CNN_TRAIN_DLL_LIBRARY``
#   Corresponding CNN training API Windows DLL. This is not set on non-Windows platforms.
#
# ``cuDNN_OPS_INFER_LINK_LIBRARY``
#   the libraries to link directly to use starndard ML operations API from cuDNN.
# ``cuDNN_OPS_INFER_DLL_LIBRARY``
#   Corresponding OPS inference API Windows DLL. This is not set on non-Windows platforms.
# ``cuDNN_OPS_TRAIN_LINK_LIBRARY``
#   the libraries to link directly to use starndard ML operations API from cuDNN.
# ``cuDNN_OPS_TRAIN_DLL_LIBRARY``
#   Corresponding OPS inference API Windows DLL. This is not set on non-Windows platforms.
#
# ``cuDNN_FOUND``
#   If false, do not try to use cuDNN.
# ``cuDNN_VERSION``
#   Version of the cuDNN library found
# ``cuDNN_VERSION_MAJOR``
#   Major Version of the cuDNN library found
# ``cuDNN_VERSION_MINOR``
#   Minor Version of the cuDNN library found

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
  set(cuDNN_VERSION_MAJOR ${CUDNN_MAJOR_VERSION})
  set(cuDNN_VERSION_MINOR ${CUDNN_MINOR_VERSION})
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

  macro(af_find_cudnn_libs cudnn_lib_name_infix)
    if("${cudnn_lib_name_infix}" STREQUAL "")
	  set(LIB_INFIX "")
	else()
	  string(TOUPPER ${cudnn_lib_name_infix} LIB_INFIX)
	endif()
    find_library(cuDNN${LIB_INFIX}_LINK_LIBRARY
      NAMES
        libcudnn${cudnn_lib_name_infix}.so.${cudnn_ver_suffix}
        libcudnn${cudnn_lib_name_infix}.${cudnn_ver_suffix}.dylib
        cudnn${cudnn_lib_name_infix}
      PATHS
        ${cuDNN_ROOT_DIR}
        ${PC_CUDNN_LIBRARY_DIRS}
        $ENV{LD_LIBRARY_PATH}
        ${libpath_cudart}
        ${CMAKE_INSTALL_PREFIX}
      PATH_SUFFIXES lib lib64 bin lib/x64 bin/x64
      DOC "cudnn${cudnn_lib_name_infix} link library." )
    mark_as_advanced(cuDNN${LIB_INFIX}_LINK_LIBRARY)

    if(WIN32 AND cuDNN_LINK_LIBRARY)
      find_file(cuDNN${LIB_INFIX}_DLL_LIBRARY
      NAMES cudnn${cudnn_lib_name_infix}64_${cudnn_ver_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
      PATHS
        ${cuDNN_ROOT_DIR}
        ${PC_CUDNN_LIBRARY_DIRS}
        $ENV{PATH}
        ${libpath_cudart}
        ${CMAKE_INSTALL_PREFIX}
      PATH_SUFFIXES lib lib64 bin lib/x64 bin/x64
      DOC "cudnn${cudnn_lib_name_infix} Windows DLL." )
      mark_as_advanced(cuDNN${LIB_INFIX}_DLL_LIBRARY)
    endif()
  endmacro()

  af_find_cudnn_libs("") # gets base cudnn shared library
  if(cuDNN_VERSION_MAJOR VERSION_GREATER 8 OR cuDNN_VERSION_MAJOR VERSION_EQUAL 8)
    af_find_cudnn_libs("_adv_infer")
    af_find_cudnn_libs("_adv_train")
    af_find_cudnn_libs("_cnn_infer")
    af_find_cudnn_libs("_cnn_train")
    af_find_cudnn_libs("_ops_infer")
    af_find_cudnn_libs("_ops_train")
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
  if(cuDNN_VERSION_MAJOR VERSION_GREATER 8 OR cuDNN_VERSION_MAJOR VERSION_EQUAL 8)
    macro(create_cudnn_target cudnn_target_name)
	  string(TOUPPER ${cudnn_target_name} target_infix)
	  add_library(cuDNN::${cudnn_target_name} SHARED IMPORTED)
	  if(WIN32)
        set_target_properties(cuDNN::${cudnn_target_name}
          PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGE "C"
          INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
          IMPORTED_LOCATION "${cuDNN_${target_infix}_DLL_LIBRARY}"
          IMPORTED_IMPLIB "${cuDNN_${target_infix}_LINK_LIBRARY}"
        )
      else(WIN32)
          set_target_properties(cuDNN::${cudnn_target_name}
            PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGE "C"
            INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${cuDNN_${target_infix}_LINK_LIBRARY}"
          )
      endif(WIN32)
	endmacro()
	create_cudnn_target(adv_infer)
	create_cudnn_target(adv_train)
	create_cudnn_target(cnn_infer)
	create_cudnn_target(cnn_train)
	create_cudnn_target(ops_infer)
	create_cudnn_target(ops_train)
  endif()
endif(cuDNN_FOUND)
