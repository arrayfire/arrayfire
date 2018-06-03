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

find_package(PkgConfig)
pkg_check_modules(PKG_FFTW "fftw3")

find_path( FFTW_INCLUDE_DIR
  NAMES "fftw3.h"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_INCLUDE_PATH}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${PKG_FFTW_INCLUDE_DIRS}
  PATH_SUFFIXES "include" "include/fftw"
  )

find_library( FFTW_LIBRARY
  NAMES "fftw3" "libfftw3-3" "fftw3-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

find_library( FFTWF_LIBRARY
  NAMES "fftw3f" "libfftw3f-3" "fftw3f-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${CMAKE_SYSTEM_LIBRARY_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARY FFTWF_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
    FFTW_INCLUDE_DIR FFTW_LIBRARY FFTWF_LIBRARY)

if (FFTW_FOUND)
  add_library(FFTW::FFTW UNKNOWN IMPORTED)
  set_target_properties(FFTW::FFTW PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGE "C"
    IMPORTED_LOCATION "${FFTW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")

  add_library(FFTW::FFTWF UNKNOWN IMPORTED)
  set_target_properties(FFTW::FFTWF PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGE "C"
    IMPORTED_LOCATION "${FFTWF_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")
endif (FFTW_FOUND)

