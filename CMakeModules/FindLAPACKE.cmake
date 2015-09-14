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

IF(NOT LAPACKE_ROOT AND ENV{LAPACKEDIR})
  SET(LAPACKE_ROOT $ENV{LAPACKEDIR})
ENDIF()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT LAPACKE_ROOT)
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

    IF(LAPACKE_ROOT)
        #find libs
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "lapacke" "LAPACKE" "liblapacke"
            PATHS ${LAPACKE_ROOT}
            PATH_SUFFIXES "lib" "lib64"
            DOC "LAPACKE Library"
            NO_DEFAULT_PATH
            )
        FIND_LIBRARY(
            LAPACK_LIB
            NAMES "lapack" "LAPACK" "liblapack"
            PATHS ${LAPACKE_ROOT}
            PATH_SUFFIXES "lib" "lib64"
            DOC "LAPACK Library"
            NO_DEFAULT_PATH
            )
        FIND_PATH(
            LAPACKE_INCLUDES
            NAMES "lapacke.h"
            PATHS ${LAPACKE_ROOT}
            PATH_SUFFIXES "include"
            DOC "LAPACKE Include Directory"
            NO_DEFAULT_PATH
            )

    ELSE()
        FIND_LIBRARY(
            LAPACKE_LIB
            NAMES "lapacke" "liblapacke"
            PATHS
            ${PC_LAPACKE_LIBRARY_DIRS}
            ${LIB_INSTALL_DIR}
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
            NAMES "lapack" "liblapack"
            PATHS
            ${PC_LAPACKE_LIBRARY_DIRS}
            ${LIB_INSTALL_DIR}
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
    ENDIF(LAPACKE_ROOT)
ENDIF(PC_LAPACKE_FOUND)

SET(LAPACK_LIBRARIES ${LAPACKE_LIB} ${LAPACK_LIB})
SET(LAPACK_INCLUDE_DIR ${LAPACKE_INCLUDES})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LAPACK DEFAULT_MSG
  LAPACK_INCLUDE_DIR LAPACK_LIBRARIES)

MARK_AS_ADVANCED(LAPACK_INCLUDES LAPACK_LIBRARIES)
