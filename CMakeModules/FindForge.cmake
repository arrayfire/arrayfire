# - Find Forge

# Defines the following variables:
# FORGE_INCLUDE_DIRECTORIES    - Location of FORGE's include directory.
# FORGE_LIBRARIES              - Location of FORGE's libraries.
# FORGE_FOUND                  - True if FORGE has been located
#
# You may provide a hint to where FORGE's root directory may be located
# by setting FORGE_ROOT_DIR before calling this script.
#
# ----------------------------------------------------------------------------
#
# FORGE_FOUND        - True of the FORGE library has been found.
# FORGE_LIBRARIES    - Location of FORGE library, if found
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

IF(FORGE_INCLUDE_DIRECTORIES)
  # Already in cache, be silent
  set (FORGE_FIND_QUIETLY TRUE)
ENDIF()

# Find the FORGE install directories and headers:
FIND_PATH(FORGE_ROOT_DIR
    NAMES include/forge.h
    PATH_SUFFIXES forge FORGE FORGE
    HINTS "${CMAKE_INSTALL_PREFIX}" "${FORGE_ROOT_DIR}" "${FORGE_ROOT_DIR}/lib" "${FORGE_ROOT_DIR}/lib64" "${CMAKE_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/../.."
    DOC "FORGE root directory.")

FIND_PATH(FORGE_PACKAGE_DIR
    NAMES include/forge.h lib
    HINTS "${FORGE_ROOT_DIR}/package" "${FORGE_ROOT_DIR}/build/package" "${FORGE_ROOT_DIR}"
    DOC "FORGE Package directory.")

FIND_PATH(FORGE_INCLUDE_DIRECTORIES
    NAMES forge.h
    HINTS "${FORGE_PACKAGE_DIR}/include"
    DOC "FORGE Include directory")

# Find all libraries required for the FORGE
FIND_LIBRARY(FORGE_LIBRARY
    NAMES forge
    HINTS "${FORGE_PACKAGE_DIR}/lib")

INCLUDE("${CMAKE_MODULE_PATH}/FindGLEWmx.cmake")

IF(GLEWmx_FOUND AND OPENGL_FOUND)
    IF(FORGE_INCLUDE_DIRECTORIES)
        SET(FORGE_INCLUDE_DIRECTORIES ${FORGE_INCLUDE_DIRECTORIES} ${GLEW_INCLUDE_DIR}
            CACHE INTERNAL "All include dirs required for FORGE'")
    ENDIF()
    IF(FORGE_LIBRARY)
        SET(FORGE_LIBRARIES ${FORGE_LIBRARY} ${GLEWmx_LIBRARY} ${OPENGL_gl_LIBRARY}
            CACHE INTERNAL "All libraries required for FORGE'")
    ENDIF()
    # handle the QUIETLY and REQUIRED arguments and set FORGE_FOUND to TRUE if
    # all listed variables are TRUE
    INCLUDE (FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(FORGE DEFAULT_MSG FORGE_LIBRARIES FORGE_INCLUDE_DIRECTORIES)
    MARK_AS_ADVANCED(FORGE_LIBRARIES FORGE_INCLUDE_DIRECTORIES)

ELSE(GLEWmx_FOUND AND OPENGL_FOUND)
    IF(NOT GLEWmx_FOUND)
        MESSAGE(FATAL_ERROR "GLEW-MX Not Found")
    ELSEIF(NOT OPENGL_FOUND)
        MESSAGE(FATAL_ERROR "OpenGL Not Found")
    ENDIF()
ENDIF(GLEWmx_FOUND AND OPENGL_FOUND)
