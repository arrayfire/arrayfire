# - Find the FFTW library
#
# Usage:
#   FIND_PACKAGE(FFTW [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#
#If environment variable FFTWDIR is specified, it has same effect as FFTW_ROOT

######## This FindFFTW.cmake file is a copy of the file from the eigen library
######## http://code.metager.de/source/xref/lib/eigen/cmake/FindFFTW.cmake

IF(NOT FFTW_ROOT AND ENV{FFTWDIR})
    SET(FFTW_ROOT $ENV{FFTWDIR})
ENDIF()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT FFTW_ROOT)
    PKG_CHECK_MODULES( PKG_FFTW QUIET "fftw3")
ENDIF()

#Check whether to search static or dynamic libs
SET(CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES})
IF(${FFTW_USE_STATIC_LIBS} )
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
ELSE()
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
ENDIF()

IF(FFTW_ROOT)
    #find libs
    FIND_LIBRARY(
        FFTW_LIB
        NAMES "fftw3" "libfftw3-3" "fftw3-3"
        PATHS ${FFTW_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )
    FIND_LIBRARY(
        FFTWF_LIB
        NAMES "fftw3f" "libfftw3f-3" "fftw3f-3"
        PATHS ${FFTW_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    #find includes
    FIND_PATH(
        FFTW_INCLUDES
        NAMES "fftw3.h"
        PATHS ${FFTW_ROOT}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH
        )
ELSE()
    FIND_LIBRARY(
        FFTW_LIB
        NAMES "fftw3"
        PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )
    FIND_LIBRARY(
        FFTWF_LIB
        NAMES "fftw3f"
        PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )
    FIND_PATH(
        FFTW_INCLUDES
        NAMES "fftw3.h"
        PATHS ${PKG_FFTW_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
        )
ENDIF(FFTW_ROOT)

SET(FFTW_LIBRARIES ${FFTW_LIB} ${FFTWF_LIB})

SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG
    FFTW_INCLUDES FFTW_LIBRARIES)

MARK_AS_ADVANCED(FFTW_INCLUDES FFTW_LIBRARIES)
