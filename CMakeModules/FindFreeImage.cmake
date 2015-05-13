#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGE_FOUND
# FREEIMAGE_INCLUDE_PATH
# FREEIMAGE_LIBRARY
#

OPTION(FREEIMAGE_STATIC "Use Static FreeImage Lib" OFF)

FIND_PATH( FREEIMAGE_INCLUDE_PATH
    NAMES FreeImage.h
    HINTS ${PROJECT_SOURCE_DIR}/extern/FreeImage
    PATHS
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    DOC "The directory where FreeImage.h resides")

FIND_LIBRARY( FREEIMAGE_DYNAMIC_LIBRARY
    NAMES FreeImage freeimage
    HINTS ${PROJECT_SOURCE_DIR}/FreeImage
    PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /sw/lib
    /opt/local/lib
    DOC "The FreeImage library")

SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
FIND_LIBRARY( FREEIMAGE_STATIC_LIBRARY
    NAMES FreeImageLIB FreeImage freeimage
    HINTS ${PROJECT_SOURCE_DIR}/FreeImage
    PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /sw/lib
    /opt/local/lib
    DOC "The FreeImage library")

IF(FREEIMAGE_STATIC)
  MESSAGE(STATUS "Using Static FreeImage Lib")
  ADD_DEFINITIONS(-DFREEIMAGE_LIB)
  SET(FREEIMAGE_LIBRARY ${FREEIMAGE_STATIC_LIBRARY})
ELSE(FREEIMAGE_STATIC)
  MESSAGE(STATUS "Using Dynamic FreeImage Lib")
  SET(FREEIMAGE_LIBRARY ${FREEIMAGE_DYNAMIC_LIBRARY})
ENDIF(FREEIMAGE_STATIC)

MARK_AS_ADVANCED(
    FREEIMAGE_LIBRARY
    FREEIMAGE_INCLUDE_PATH)
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG
    FREEIMAGE_INCLUDE_PATH FREEIMAGE_LIBRARY)
