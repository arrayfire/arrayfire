# Sourced from
#https://code.google.com/p/assembly3d/

# Locate the glfw library
# This module defines the following variables:
# GLFW_LIBRARY, the name of the library;
# GLFW_INCLUDE_DIR, where to find glfw include files.
# GLFW_FOUND, true if both the GLFW_LIBRARY and GLFW_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you could define an environment variable called
# GLFW_ROOT which points to the root of the glfw library installation. This is pretty useful
# on a Windows platform.
#
#
# Usage example to compile an "executable" target to the glfw library:
#
# FIND_PACKAGE (glfw REQUIRED)
# INCLUDE_DIRECTORIES (${GLFW_INCLUDE_DIR})
# ADD_EXECUTABLE (executable ${EXECUTABLE_SRCS})
# TARGET_LINK_LIBRARIES (executable ${GLFW_LIBRARY})
#
# TODO:
# Allow the user to select to link to a shared library or to a static library.

#Search for the include file...
FIND_PATH(GLFW_INCLUDE_DIR GLFW/glfw3.h DOC "Path to GLFW include directory."
  HINTS
  $ENV{GLFW_ROOT}
  PATH_SUFFIX include #For finding the include file under the root of the glfw expanded archive, typically on Windows.
  PATHS
  /usr/include/
  /usr/local/include/
  # By default headers are under GLFW subfolder
  /usr/include/GLFW
  /usr/local/include/GLFW
  ${GLFW_ROOT_DIR}/include/ # added by ptr
)

FIND_LIBRARY(GLFW_LIBRARY DOC "Absolute path to GLFW library."
  NAMES glfw GLFW.lib glfw3
  HINTS
  $ENV{GLFW_ROOT}
  PATH_SUFFIXES lib/win32 #For finding the library file under the root of the glfw expanded archive, typically on Windows.
  PATHS
  /usr/lib
  /usr/lib64
  /usr/lib/x86_64-linux-gnu
  /usr/lib/arm-linux-gnueabihf
  /usr/local/lib
  /usr/local/lib64
  /sw/lib
  /opt/local/lib
  ${GLFW_ROOT_DIR}/lib-msvc100/release # added by ptr
)

SET(GLFW_FOUND 0)
IF(GLFW_LIBRARY AND GLFW_INCLUDE_DIR)
  SET(GLFW_FOUND 1)
  message(STATUS "GLFW found!")
ENDIF(GLFW_LIBRARY AND GLFW_INCLUDE_DIR)
