# FindFreeImage.cmake
# Author: Umar Arshad <umar@arrayfire.com>
#
# Finds the FreeImage libraries
# Sets the following variables:
#          FreeImage_FOUND
#          FreeImage_INCLUDE_DIR
#          FreeImage_DYNAMIC_LIBRARY
#          FreeImage_STATIC_LIBRARY
#
# Usage:
# find_package(FreeImage)
# if (FreeImage_FOUND)
#    target_link_libraries(mylib PRIVATE FreeImage::FreeImage)
# endif (FreeImage_FOUND)
#
# OR if you want to link against the static library:
#
# find_package(FreeImage)
# if (FreeImage_FOUND)
#    target_link_libraries(mylib PRIVATE FreeImage::FreeImage_STATIC)
# endif (FreeImage_FOUND)
#
# NOTE: You do not need to include the FreeImage include directories since they
# will be included as part of the target_link_libraries command

find_path( FreeImage_INCLUDE_DIR
    NAMES FreeImage.h
    HINTS ${PROJECT_SOURCE_DIR}/extern/FreeImage
    PATHS
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    DOC "The directory where FreeImage.h resides")

find_library( FreeImage_DYNAMIC_LIBRARY
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

find_library( FreeImage_STATIC_LIBRARY
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

mark_as_advanced(
    FreeImage_INCLUDE_DIR
    FreeImage_DYNAMIC_LIBRARY
    FreeImage_STATIC_LIBRARY
    )
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FreeImage
  REQUIRED_VARS FreeImage_INCLUDE_DIR FreeImage_DYNAMIC_LIBRARY
  )

set(FREEIMAGE_LIBRARY ${FreeImage_DYNAMIC_LIBRARY})

if (FreeImage_FOUND AND NOT TARGET FreeImage::FreeImage)
  add_library(FreeImage::FreeImage UNKNOWN IMPORTED)
  set_target_properties(FreeImage::FreeImage PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGE "C"
    IMPORTED_LOCATION "${FreeImage_DYNAMIC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}")

  if(FreeImage_STATIC_LIBRARY_FOUND)
    add_library(FreeImage::FreeImage_STATIC UNKNOWN IMPORTED)
    set_target_properties(FreeImage::FreeImage_STATIC PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      IMPORTED_LOCATION "${FreeImage_STATIC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}")
  endif(FreeImage_STATIC_LIBRARY_FOUND)
endif()
