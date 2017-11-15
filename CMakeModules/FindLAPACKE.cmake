# - Find the LAPACKE library
#
# Usage:
#   FIND_PACKAGE(LAPACKE [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   LAPACK_FOUND               ... true if LAPACKE is found on the system
#   LAPACK_LIBRARIES           ... full path to LAPACKE library
#   LAPACK_INCLUDES            ... LAPACKE include directory
#

SET(LAPACKE_ROOT_DIR "" CACHE STRING
  "Root directory for custom LAPACK implementation")

IF (NOT INTEL_MKL_ROOT_DIR)
  SET(INTEL_MKL_ROOT_DIR $ENV{INTEL_MKL_ROOT})
ENDIF()

IF(NOT LAPACKE_ROOT_DIR)

  IF (ENV{LAPACKEDIR})
    SET(LAPACKE_ROOT_DIR $ENV{LAPACKEDIR})
  ENDIF()

  IF (ENV{LAPACKE_ROOT_DIR})
    SET(LAPACKE_ROOT_DIR $ENV{LAPACKE_ROOT_DIR})
  ENDIF()

  IF (INTEL_MKL_ROOT_DIR)
    SET(LAPACKE_ROOT_DIR ${INTEL_MKL_ROOT_DIR})
  ENDIF()
ENDIF()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT LAPACKE_ROOT_DIR)
  PKG_CHECK_MODULES( PC_LAPACKE QUIET "lapacke")
ENDIF()

IF(PC_LAPACKE_FOUND)
    FOREACH(PC_LIB ${PC_LAPACKE_LIBRARIES})
      FIND_LIBRARY(${PC_LIB}_LIBRARY NAMES ${PC_LIB} HINTS ${PC_LAPACKE_LIBRARY_DIRS} )
      IF (NOT ${PC_LIB}_LIBRARY)
        MESSAGE(FATAL_ERROR "Something is wrong in your pkg-config file - lib ${PC_LIB} not found in ${PC_LAPACKE_LIBRARY_DIRS}")
      ENDIF (NOT ${PC_LIB}_LIBRARY)
      LIST(APPEND LAPACKE_LIB ${${PC_LIB}_LIBRARY})
    ENDFOREACH(PC_LIB)

    FIND_PATH(
        LAPACKE_INCLUDES
        NAMES "lapacke.h"
        PATHS
        ${PC_LAPACKE_INCLUDE_DIRS}
        ${INCLUDE_INSTALL_DIR}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        DOC "LAPACKE Include Directory"
        )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LAPACKE DEFAULT_MSG LAPACKE_LIB)
    MARK_AS_ADVANCED(LAPACKE_INCLUDES LAPACKE_LIB)

ELSE(PC_LAPACKE_FOUND)

    IF ("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
        SET(MKL_LIB_DIR_SUFFIX "intel64")
    ELSE()
        SET(MKL_LIB_DIR_SUFFIX "ia32")
    ENDIF()

    IF(LAPACKE_ROOT_DIR)
        #find libs
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "lapacke" "LAPACKE" "liblapacke" "mkl_rt"
            PATHS ${LAPACKE_ROOT_DIR}
            PATH_SUFFIXES "lib" "lib64" "lib/${MKL_LIB_DIR_SUFFIX}"
            DOC "LAPACKE Library"
            NO_DEFAULT_PATH
            )
        FIND_LIBRARY(
            LAPACK_LIB
            NAMES "lapack" "LAPACK" "liblapack" "mkl_rt"
            PATHS ${LAPACKE_ROOT_DIR}
            PATH_SUFFIXES "lib" "lib64" "lib/${MKL_LIB_DIR_SUFFIX}"
            DOC "LAPACK Library"
            NO_DEFAULT_PATH
            )
        FIND_PATH(
            LAPACKE_INCLUDES
            NAMES "lapacke.h" "mkl_lapacke.h"
            PATHS ${LAPACKE_ROOT_DIR}
            PATH_SUFFIXES "include"
            DOC "LAPACKE Include Directory"
            NO_DEFAULT_PATH
            )
    ELSE()
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "mkl_rt" "lapacke" "liblapacke" "openblas"
            PATHS
            ${PC_LAPACKE_LIBRARY_DIRS}
            ${LIB_INSTALL_DIR}
            /opt/intel/mkl/lib/${MKL_LIB_DIR_SUFFIX}
            /usr/lib64
            /usr/lib
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            DOC "LAPACKE Library"
            )
        FIND_LIBRARY(
           LAPACK_LIB
            NAMES "mkl_rt" "lapack" "liblapack" "openblas"
            PATHS
            ${PC_LAPACKE_LIBRARY_DIRS}
            ${LIB_INSTALL_DIR}
            /opt/intel/mkl/lib/${MKL_LIB_DIR_SUFFIX}
            /usr/lib64
            /usr/lib
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            DOC "LAPACK Library"
            )
        FIND_PATH(
            LAPACKE_INCLUDES
            NAMES "mkl_lapacke.h" "lapacke.h"
            PATHS
            ${PC_LAPACKE_INCLUDE_DIRS}
            ${INCLUDE_INSTALL_DIR}
            /opt/intel/mkl/include
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "LAPACKE Include Directory"
            PATH_SUFFIXES
            lapacke
            )
    ENDIF(LAPACKE_ROOT_DIR)
ENDIF(PC_LAPACKE_FOUND)

IF(PC_LAPACKE_FOUND OR (LAPACKE_LIB AND LAPACK_LIB))
    SET(LAPACK_LIBRARIES ${LAPACKE_LIB} ${LAPACK_LIB})
ENDIF()
IF(LAPACKE_INCLUDES)
    SET(LAPACK_INCLUDE_DIR ${LAPACKE_INCLUDES})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LAPACK DEFAULT_MSG
  LAPACK_INCLUDE_DIR LAPACK_LIBRARIES)

MARK_AS_ADVANCED(
  LAPACKE_ROOT_DIR
  LAPACK_INCLUDES
  LAPACK_LIBRARIES
  LAPACK_LIB
  LAPACKE_INCLUDES
  LAPACKE_LIB)
