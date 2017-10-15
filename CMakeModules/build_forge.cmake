# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

set(FORGE_VERSION 1.0.2-ft)
set(prefix "${ArrayFire_BINARY_DIR}/third_party/forge")

if(MSVC)
  set(disable_warning_flags "/wd4251")
  set(forge_shared_lib "${ArrayFire_BINARY_DIR}/third_party/forge/lib/${CMAKE_SHARED_LIBRARY_PREFIX}forge${CMAKE_LINK_LIBRARY_SUFFIX}")
else()
  set(forge_shared_lib "${ArrayFire_BINARY_DIR}/third_party/forge/lib/${CMAKE_SHARED_LIBRARY_PREFIX}forge${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# FIXME Tag forge correctly during release
ExternalProject_Add(
    forge-ext
    GIT_REPOSITORY https://github.com/arrayfire/forge.git
    GIT_TAG v${FORGE_VERSION}
    PREFIX "${prefix}"
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS ${forge_shared_lib}
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_DOCUMENTATION:BOOL=OFF
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_FLAGS:STRING=${disable_warning_flags}
      -Dglbinding_DIR:STRING=${glbinding_DIR}
      -DGLFW_ROOT_DIR:STRING=${GLFW_ROOT_DIR}
      -DBOOST_INCLUDEDIR:PATH=${Boost_INCLUDE_DIRS}
      -Dglbinding_DIR:PATH=${glbinding_DIR}
      -DUSE_SYSTEM_GLBINDING:BOOL=TRUE
      -DUSE_FREEIMAGE:BOOL=OFF
    )

# NOTE: This approach doesn't work because the ExternalProject_Add outputs are
# created at build time. The targets are created at configuration time.
#
# make_directory("${prefix}/include")
# make_directory("${ArrayFire_BINARY_DIR}/third_party/forge/lib")
# execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${forge_shared_lib}")

# add_library(Forge::Forge SHARED IMPORTED GLOBAL)
# set_target_properties(Forge::Forge PROPERTIES
#   INTERFACE_LINK_LIBRARIES "${forge_shared_lib}"
#   INTERFACE_INCLUDE_DIRECTORIES "${prefix}/include"
#   )
#
# add_dependencies(Forge::Forge forge-ext)

set(Forge_INCLUDE_DIR "${prefix}/include")
set(Forge_LIBRARIES "${forge_shared_lib}")

find_package_handle_standard_args(Forge DEFAULT_MSG
    Forge_INCLUDE_DIR Forge_LIBRARIES)
