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

if(NOT TARGET OpenCL::cl2hpp)
  find_path(cl2hpp_header_file_path
    NAMES CL/cl2.hpp
    PATHS ${OpenCL_INCLUDE_PATHS})

  if(cl2hpp_header_file_path)
    add_library(cl2hpp IMPORTED INTERFACE GLOBAL)
    add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)

    set_target_properties(cl2hpp OpenCL::cl2hpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${cl2hpp_header_file_path})
  elseif (NOT TARGET OpenCL::cl2hpp OR NOT TARGET cl2hpp)
    af_dep_check_and_populate(${cl2hpp_prefix}
      URI https://github.com/KhronosGroup/OpenCL-CLHPP.git
      REF v2022.09.30)

    find_path(cl2hpp_var
      NAMES CL/cl2.hpp
      PATHS ${ArrayFire_BINARY_DIR}/extern/${cl2hpp_prefix}-src/include)

    add_library(cl2hpp IMPORTED INTERFACE GLOBAL)
    add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)

    set_target_properties(cl2hpp OpenCL::cl2hpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${cl2hpp_var})
  endif()
endif()
