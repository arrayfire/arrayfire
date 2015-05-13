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
IF(FREEIMAGE_STATIC)
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
  FIND_LIBRARY( FREEIMAGE_LIBRARY
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
  ADD_DEFINITIONS(-DFREEIMAGE_LIB)
ELSE(FREEIMAGE_STATIC)
FIND_LIBRARY( FREEIMAGE_LIBRARY
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
ENDIF(FREEIMAGE_STATIC)

SET(FREEIMAGE_INCLUDE_DIRS ${FREEIMAGE_INCLUDE_PATH})
SET(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})

MARK_AS_ADVANCED(
    FREEIMAGE_LIBRARY
    FREEIMAGE_INCLUDE_PATH)
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG
    FREEIMAGE_INCLUDE_PATH FREEIMAGE_LIBRARY)
