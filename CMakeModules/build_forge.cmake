# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

set(FORGE_VERSION af3.6.0)
set(prefix "${ArrayFire_BINARY_DIR}/third_party/forge")
set(PX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(SX ${CMAKE_SHARED_LIBRARY_SUFFIX})

if(MSVC)
  set(disable_warning_flags "/wd4251")
  set(SX ${CMAKE_LINK_LIBRARY_SUFFIX})
endif()

set(forge_lib "${PROJECT_BINARY_DIR}/third_party/forge/lib/${PX}forge${SX}")

# Create a list with an alternate separator e.g. pipe symbol
string(REPLACE ";" "|" CMAKE_PREFIX_PATH_ALT_SEP "${CMAKE_PREFIX_PATH}")

# FIXME Tag forge correctly during release
ExternalProject_Add(
    forge-ext
    GIT_REPOSITORY https://github.com/arrayfire/forge.git
    GIT_TAG ${FORGE_VERSION}
    PREFIX "${prefix}"
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS ${forge_lib}
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
	LIST_SEPARATOR | # Use the alternate list separator
    CMAKE_ARGS
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH_ALT_SEP}"
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DCMAKE_CXX_FLAGS:STRING=${disable_warning_flags}
      -DFG_BUILD_EXAMPLES:BOOL=OFF
      -DFG_BUILD_DOCS:BOOL=OFF
      -DFG_WITH_FREEIMAGE:BOOL=OFF
      -DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}
    )

# NOTE: This approach doesn't work because the ExternalProject_Add outputs are
# created at build time. The targets are created at configuration time.
#
# make_directory("${prefix}/include")
# make_directory("${ArrayFire_BINARY_DIR}/third_party/forge/lib")
# execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${forge_lib}")

# add_library(Forge::Forge SHARED IMPORTED GLOBAL)
# set_target_properties(Forge::Forge PROPERTIES
#   INTERFACE_LINK_LIBRARIES "${forge_lib}"
#   INTERFACE_INCLUDE_DIRECTORIES "${prefix}/include"
#   )
#
# add_dependencies(Forge::Forge forge-ext)

set(Forge_INCLUDE_DIR "${prefix}/include")
set(Forge_LIBRARIES "${forge_lib}")

find_package_handle_standard_args(Forge DEFAULT_MSG
    Forge_INCLUDE_DIR Forge_LIBRARIES)
