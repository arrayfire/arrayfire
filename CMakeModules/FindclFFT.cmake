# - Find clFFT, AMD's OpenCL FFT library

# This script defines the following variables:
# CLFFT_INCLUDE_DIRS    - Location of clFFT's include directory.
# CLFFT_LIBRARIES       - Location of clFFT's libraries
# CLFFT_FOUND           - True if clFFT has been located
#
# If your clFFT installation is not in a standard installation directory, you
# may provide a hint to where it may be found. Simply set the value CLFFT_ROOT
# to the directory containing 'include/clFFT.h" prior to calling this script.
#
# By default this script will attempt to find the 32-bit version of clFFT.
# If you desire to use the 64-bit version instead, set
#   set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
# prior to calling this script.
#
#=============================================================================
# Copyright 2014 Brian Kloppenborg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

IF(CLFFT_INCLUDE_DIRS)
    # Already in cache, be silent
    set (CLFFT_FIND_QUIETLY TRUE)
ENDIF (CLFFT_INCLUDE_DIRS)

FIND_PATH(CLFFT_SOURCE_DIR
    NAMES src/include/clFFT.h
    HINTS   "${CMAKE_SOURCE_DIR}/clFFT"
            "${CMAKE_SOURCE_DIR}/../clFFT"
            "${CMAKE_SOURCE_DIR}/../../clFFT"
    DOC "clFFT source directory.")

FIND_PATH(CLFFT_BUILD_DIR
    NAMES ./install_manifest.txt
    HINTS   "${CLFFT_SOURCE_DIR}/build"
            "${CLFFT_SOURCE_DIR}/src/build"
    DOC "clFFT build directory.")

FIND_PATH(CLFFT_ROOT_DIR
    NAMES include/clFFT.h
    HINTS   "$ENV{CLFFT_ROOT}"
            "${CMAKE_SOURCE_DIR}/.."
            "${CMAKE_SOURCE_DIR}/../.."
            "${CLFFT_BUILD_DIR}/package"
            "${CMAKE_INSTALL_PREFIX}"
    DOC "clFFT root directory.")

FIND_PATH(_CLFFT_INCLUDE_DIRS
    NAMES clFFT.h
    HINTS "${CLFFT_ROOT_DIR}/include"
    DOC "clFFT Include directory")

FIND_LIBRARY(_CLFFT_LIBRARY
    NAMES clFFT
    HINTS 	"${CLFFT_ROOT_DIR}/lib64"
		    "${CLFFT_ROOT_DIR}/lib64/import"
		    "${CMAKE_INSTALL_PREFIX}")

SET(CLFFT_INCLUDE_DIRS ${_CLFFT_INCLUDE_DIRS})
SET(CLFFT_LIBRARIES ${_CLFFT_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set CLFFT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLFFT DEFAULT_MSG CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)
MARK_AS_ADVANCED(CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)
