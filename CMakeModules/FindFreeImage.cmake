#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGE_FOUND
# FREEIMAGE_INCLUDE_PATH
# FREEIMAGE_LIBRARY
# FREEIMAGE_STATIC_LIBRARY
# FREEIMAGE_DYNAMIC_LIBRARY
#

OPTION(USE_FREEIMAGE_STATIC "Use Static FreeImage Lib" OFF)

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

SET(PX ${CMAKE_STATIC_LIBRARY_PREFIX})
SET(SX ${CMAKE_STATIC_LIBRARY_SUFFIX})
FIND_LIBRARY( FREEIMAGE_STATIC_LIBRARY
    NAMES ${PX}FreeImageLIB${SX} ${PX}FreeImage${SX} ${PX}freeimage${SX}
    HINTS ${PROJECT_SOURCE_DIR}/FreeImage
    PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /sw/lib
    /opt/local/lib
    DOC "The FreeImage library")
UNSET(PX)
UNSET(SX)

IF(USE_FREEIMAGE_STATIC)
  MESSAGE(STATUS "Using Static FreeImage Lib")
  ADD_DEFINITIONS(-DFREEIMAGE_LIB)
  SET(FREEIMAGE_LIBRARY ${FREEIMAGE_STATIC_LIBRARY})
ELSE(USE_FREEIMAGE_STATIC)
  MESSAGE(STATUS "Using Dynamic FreeImage Lib")
  REMOVE_DEFINITIONS(-DFREEIMAGE_LIB)
  SET(FREEIMAGE_LIBRARY ${FREEIMAGE_DYNAMIC_LIBRARY})
ENDIF(USE_FREEIMAGE_STATIC)

MARK_AS_ADVANCED(
    FREEIMAGE_DYNAMIC_LIBRARY
    FREEIMAGE_STATIC_LIBRARY
    )
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG
    FREEIMAGE_INCLUDE_PATH FREEIMAGE_LIBRARY)
