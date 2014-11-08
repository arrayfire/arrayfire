# - Find clBLAS, AMD's OpenCL BLAS library

# This script defines the standard variables
#  CLBLAS_FOUND           - Whether or not clBLAS was located
#  CLBLAS_INCLUDE_DIRS    - All include directories for clBLAS headers
#  CLBLAS_LIBRARIES       - All libraries for clBLAS
#
# This script also creates a few non-standard variables that may be useful
# in your project:
#
#  CLBLAS_SOURCE_DIR      - The location of the clBLAS src directory, if found.
#  CLBLAS_PACKAGE_DIR     - The location of the clBLAS package directory, if found.
#
# If your clBLAS installation is not in a standard installation directory, you
# may provide a hint to where it may be found. Simply set the value CLBLAS_ROOT
# to the directory containing 'include/clBLAS.h" prior to calling this script.
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

# Find packages on which clBLAS depends
find_package(OpenCL REQUIRED)

# Set the CLBLAS_ROOT_DIR relative to the current directory.
IF(NOT DEFINED ${CLBLAS_ROOT_DIR})
    LIST(APPEND CLBLAS_ROOT_DIR ${CMAKE_SOURCE_DIR}/.. ${CMAKE_SOURCE_DIR}/../..)
ENDIF()

FIND_PATH(CLBLAS_SOURCE_DIR
    NAMES           src/clBlas.h src/clAmdBlas.h
    PATH_SUFFIXES   clblas clBLAS clBlas CLBLAS
    DOC             "Location of the clBLAS source directory"
    HINTS           ${CLBLAS_ROOT_DIR}
    NO_DEFAULT_PATH)

FIND_PATH(CLBLAS_PACKAGE_DIR
    NAMES           bin include lib64
    PATH_SUFFIXES   build/package
    DOC             "Location of the clBLAS install/package directory."
    HINTS           ${CLBLAS_SOURCE_DIR}
    NO_DEFAULT_PATH)
    
FIND_PATH(_CLBLAS_INCLUDE_DIR
    NAMES           clBLAS.h
    DOC             "Location of the clBLAS include directory."
    PATH_SUFFIXES   include package/include
    HINTS           /usr/local
                    ${CLBLAS_PACKAGE_DIR})
            
FIND_PATH(CLBLAS_LIBRARY_DIR
    NAMES           libclBLAS${CMAKE_SHARED_LIBRARY_SUFFIX}
    DOC             "Location of the clBLAS library"
    PATH_SUFFIXES   lib64 package/lib64 lib64/import package/lib64/import
    HINTS           ${CLBLAS_PACKAGE_DIR})

FIND_LIBRARY(_CLBLAS_LIBRARY
    NAMES           clBLAS
    DOC             "Library files"
    PATH_SUFFIXES   lib lib64 package/lib64 lib64/import package/lib64/import
    HINTS           /usr/local
                    ${CLBLAS_PACKAGE_DIR})

# Set up the includes and library directories
SET(CLBLAS_LIBRARY ${_CLBLAS_LIBRARY})
SET(CLBLAS_INCLUDE_DIRS ${_CLBLAS_INCLUDE_DIR} ${OPENCL_CL_INCLUDE_DIRS})
SET(CLBLAS_LIBRARIES ${_CLBLAS_LIBRARY} ${OPENCL_LIBRARIES})
SET(CLBLAS_SOURCE_DIR ${CLBLAS_SOURCE_DIR} 
    CACHE PATH "Path for clBLAS source, if found")
SET(CLBLAS_PACKAGE_DIR ${CLBLAS_PACKAGE_DIR} 
    CACHE PATH "Path for clBLAS's packaging directory, if found")
SET(CLBLAS_LIBRARY_DIR ${CLBLAS_LIBRARY_DIR} 
    CACHE PATH "Path for clBLAS's packaging library directory, if found")

# handle the QUIETLY and REQUIRED arguments and set CLBLAS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLBLAS DEFAULT_MSG CLBLAS_LIBRARY 
    CLBLAS_INCLUDE_DIRS CLBLAS_LIBRARIES)
MARK_AS_ADVANCED(CLBLAS_FOUND CLBLAS_PACKAGE_DIR CLBLAS_INCLUDE_DIRS
    CLBLAS_LIBRARY_DIR CLBLAS_LIBRARIES)
