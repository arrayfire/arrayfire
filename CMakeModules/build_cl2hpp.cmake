# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

# Check if cl2.hpp exsists and if not download it from khronos GitHub repo
#
# NOTE: This file does not use ExternalProject_Add because that command was
#       was not able to download files that are not archives before CMake
#       version 3.6

find_package(OpenCL)

FetchContent_Declare(
  ${cl2hpp_prefix}
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
  GIT_TAG v2.0.12
)
FetchContent_Populate(${cl2hpp_prefix})

if (NOT TARGET OpenCL::cl2hpp OR NOT TARGET cl2hpp)
  add_library(cl2hpp IMPORTED INTERFACE GLOBAL)
  add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)

  set_target_properties(cl2hpp OpenCL::cl2hpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${${cl2hpp_prefix}_SOURCE_DIR}/include)
endif()
