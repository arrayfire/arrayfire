# Fetched the original content of this file from
# https://github.com/soumith/cudnn.torch/blob/master/cmake/FindcuDNN.cmake
# which is distributed under the OSI-approved BSD 3-Clause License.

# FindcuDNN
# -------
#
# Find cuDNN library
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIET>: as usual
#
# <EXACT> : as usual, plus we do find '5.1' version if you wanted '5'
#           (not if you wanted '5.0', as usual)
# Usage
# -----
# add_exectuable(helloworld main.cpp)
# target_link_libraries(helloworld PRIVATE nvidia::cuDNN)
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
#
# NOTE: ALWAYS call find_package(cuDNN ...) after find_package(CUDA)
#       as this find module uses from cache variables set by find
#       CUDA module

find_package(PkgConfig)
pkg_check_modules(PC_CUDNN QUIET cuDNN)

get_filename_component(libpath_cudart "${CUDA_CUDART_LIBRARY}" PATH)

# We use major only in library search as major/minor is not entirely consistent
# among platforms. Also, looking for exact minor version of .so is in general
# not a good idea. More strict enforcement of minor/patch version is done
# if/when the header file is examined.
if(cuDNN_FIND_VERSION_EXACT)
  set(cudnn_ver_suffix ".${cuDNN_FIND_VERSION_MAJOR}")
  set(cudnn_lib_win_name cudnn64_${cuDNN_FIND_VERSION_MAJOR})
else()
  set(cudnn_lib_win_name cudnn64)
endif()

find_library(cuDNN_LINK_LIBRARY
  NAMES
    libcudnn.so${cudnn_ver_suffix}
    libcudnn${cudnn_ver_suffix}.dylib
    ${cudnn_lib_win_name}
  PATHS
    $ENV{LD_LIBRARY_PATH}
    ${libpath_cudart}
    ${cuDNN_ROOT_DIR}
    ${PC_CUDNN_LIBRARY_DIRS}
    ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "cuDNN link library." )

if(WIN32 AND cuDNN_LINK_LIBRARY)
    find_file(cuDNN_DLL_LIBRARY
    NAMES ${cudnn_lib_win_name}${CMAKE_SHARED_LIBRARY_SUFFIX}
    PATHS
      $ENV{PATH}
      ${libpath_cudart}
      ${cuDNN_ROOT_DIR}
      ${PC_CUDNN_LIBRARY_DIRS}
      ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin
    DOC "cuDNN Windows DLL." )
endif()

if(cuDNN_LINK_LIBRARY)
  set(cuDNN_MAJOR_VERSION ${cuDNN_FIND_VERSION_MAJOR})
  set(cuDNN_VERSION ${cuDNN_MAJOR_VERSION})
  get_filename_component(found_cudnn_root ${cuDNN_LINK_LIBRARY} PATH)
  find_path(cuDNN_INCLUDE_DIRS
    NAMES cudnn.h
    HINTS
      ${PC_CUDNN_INCLUDE_DIRS}
      ${cuDNN_ROOT_DIR}
      ${CUDA_TOOLKIT_INCLUDE}
      ${found_cudnn_root}
    PATH_SUFFIXES include
    DOC "cuDNN include directory path." )
endif()

if(cuDNN_LINK_LIBRARY AND cuDNN_INCLUDE_DIRS)
  file(READ ${cuDNN_INCLUDE_DIRS}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
    CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
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

if(CUDNN_MAJOR_VERSION)
  ## Fixing the case where 5.1 does not fit 'exact' 5.
  if(cuDNN_FIND_VERSION_EXACT AND NOT cuDNN_FIND_VERSION_MINOR)
      if("${CUDNN_MAJOR_VERSION}" STREQUAL "${cuDNN_FIND_VERSION_MAJOR}")
      set(cuDNN_VERSION ${cuDNN_FIND_VERSION})
    endif()
  endif()
else()
  # Try to set CUDNN version from config file
  set(cuDNN_VERSION ${PC_CUDNN_CFLAGS_OTHER})
endif()

find_package_handle_standard_args(
  cuDNN
  REQUIRED_VARS cuDNN_LINK_LIBRARY cuDNN_INCLUDE_DIRS
  VERSION_VAR   cuDNN_VERSION
  )

mark_as_advanced(cuDNN_LINK_LIBRARY cuDNN_INCLUDE_DIRS cuDNN_DLL_LIBRARY)

if(cuDNN_FOUND)
  add_library(nvidia::cuDNN SHARED IMPORTED)
  if(WIN32)
    set_target_properties(nvidia::cuDNN
      PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${cuDNN_DLL_LIBRARY}"
      IMPORTED_IMPLIB "${cuDNN_LINK_LIBRARY}"
    )
  else()
    set_target_properties(nvidia::cuDNN
      PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${cuDNN_LINK_LIBRARY}"
    )
  endif()
endif()
