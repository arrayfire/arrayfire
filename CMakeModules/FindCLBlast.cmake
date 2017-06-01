# Modification of FindclBLAS from the following repo
# https://github.com/CNugteren/CLBlast

# - Find CLBlast library
#
# This module finds an installed CLBlast library.
#
# This module sets the following variables:
#  CLBLAST_FOUND - set to true if a library implementing the CBLAS interface is found
#  CLBLAST_LIBRARIES - list of libraries (using full path name) to link against to use CBLAS
#  CLBLAST_INCLUDE_DIR - path to includes

# Sets the possible install locations
set(CLBLAST_HINTS
  ${CLBLAST_ROOT}
  $ENV{CLBLAST_ROOT}
)
set(CLBLAST_PATHS
  /usr/
  /usr/local/
)

# Finds the include directories
find_path(CLBLAST_INCLUDE_DIRS
  NAMES clblast.h
  HINTS ${CLBLAS_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CLBLAST_PATHS}
  DOC "CLBlast include header clblast.h"
)
mark_as_advanced(CLBLAST_INCLUDE_DIRS)

# Finds the library
find_library(CLBLAST_LIBRARIES
  NAMES clblast
  HINTS ${CLBLAST_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CLBLAST_PATHS}
  DOC "CLBlast library"
)
mark_as_advanced(CLBLAST_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CLBLAST_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'clblast.h', install CLBlast or set CLBLAST_ROOT")
endif()
if(NOT CLBLAST_LIBRARIES)
    message(STATUS "Could NOT find CLBlast library, install it or set CLBLAST_ROOT")
endif()

# Determines whether or not CLBLast was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLBlast DEFAULT_MSG CLBLAST_INCLUDE_DIRS CLBLAST_LIBRARIES)

# ==================================================================================================
