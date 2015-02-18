# - Find ArrayFire GFX

# Defines the following variables:
# AFGFX_INCLUDE_DIRS    - Location of AFGFX's include directory.
# AFGFX_LIBRARIES       - Location of AFGFX's libraries.
# AFGFX_FOUND           - True if AFGFX has been located
#
# You may provide a hint to where AFGFX's root directory may be located
# by setting AFGFX_ROOT_DIR before calling this script.
#
# ----------------------------------------------------------------------------
#
# AFGFX_FOUND        - True of the AFGFX library has been found.
# AFGFX_LIBRARIES    - Location of AFGFX library, if found
#
# Variables used by this module, they can change the default behaviour and
# need to be set before calling find_package:
#
#=============================================================================
# Copyright (c) 2014, ArrayFire
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# * Neither the name of the ArrayFire nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

IF(AFGFX_INCLUDE_DIRS)
  # Already in cache, be silent
  set (AFGFX_FIND_QUIETLY TRUE)
ENDIF()

# Find the AFGFX install directories and headers:
FIND_PATH(AFGFX_ROOT_DIR
    NAMES include/afgfx.h
    PATH_SUFFIXES afgfx AFGFX AFGFX
    HINTS "${CMAKE_INSTALL_PREFIX}" "${AFGFX_ROOT_DIR}" "${AFGFX_ROOT_DIR}/lib" "${AFGFX_ROOT_DIR}/lib64" "${CMAKE_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/../.."
    DOC "AFGFX root directory.")

FIND_PATH(AFGFX_PACKAGE_DIR
    NAMES include/afgfx.h lib
    HINTS "${AFGFX_ROOT_DIR}/package" "${AFGFX_ROOT_DIR}/build/package" "${AFGFX_ROOT_DIR}"
    DOC "AFGFX Package directory.")

FIND_PATH(_AFGFX_INCLUDE_DIRS
    NAMES afgfx.h
    HINTS "${AFGFX_PACKAGE_DIR}/include"
    DOC "AFGFX Include directory")

# Find all libraries required for the AFGFX
FIND_LIBRARY(_AFGFX_LIBRARY
    NAMES afgfx
    HINTS "${AFGFX_PACKAGE_DIR}/lib")

INCLUDE("${CMAKE_MODULE_PATH}/FindGLEWmx.cmake")
INCLUDE("${CMAKE_MODULE_PATH}/FindGLFW.cmake")

IF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)
  SET(GRAPHICS_FOUND ON)
  ADD_DEFINITIONS(-DGLEW_MX -DWITH_GRAPHICS)
ELSE(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)
    IF(NOT GLFW_FOUND)
        MESSAGE(FATAL_ERROR "GLFW Not Found")
    ELSEIF(NOT GLEWmx_FOUND)
        MESSAGE(FATAL_ERROR "GLEW-MX Not Found")
    ELSEIF(NOT OPENGL_FOUND)
        MESSAGE(FATAL_ERROR "OpenGL Not Found")
    ENDIF()
ENDIF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)

IF(_AFGFX_INCLUDE_DIRS)
    SET(AFGFX_INCLUDE_DIRS ${_AFGFX_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR} ${GLEW_INCLUDE_DIR}
        CACHE INTERNAL "All include dirs required for AFGFX'")
ENDIF()

IF(_AFGFX_LIBRARY)
    SET(AFGFX_LIBRARIES ${_AFGFX_LIBRARY} ${GLFW_LIBRARY} ${GLEWmx_LIBRARY} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
        CACHE INTERNAL "All libraries required for AFGFX'")
    SET(AFGFX_FOUND TRUE CACHE BOOL "Whether or not AFGFX' library has been located.")
ENDIF()

# handle the QUIETLY and REQUIRED arguments and set AFGFX_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(AFGFX DEFAULT_MSG AFGFX_LIBRARIES AFGFX_INCLUDE_DIRS)
MARK_AS_ADVANCED(AFGFX_LIBRARIES AFGFX_INCLUDE_DIRS)
