# Copyright (c) 2017, ArrayFire
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

set(cl2hpp_file_url "https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp")
set(cl2hpp_file "${ArrayFire_BINARY_DIR}/include/CL/cl2.hpp")

if(OpenCL_FOUND)
  if (NOT EXISTS ${cl2hpp_file})
      message(STATUS "Downloading ${cl2hpp_file_url}")
      file(DOWNLOAD ${cl2hpp_file_url} ${cl2hpp_file}
        EXPECTED_HASH MD5=c38d1b78cd98cc809fa2a49dbd1734a5)
  endif()
  get_filename_component(download_dir ${cl2hpp_file} DIRECTORY)

  if (NOT TARGET OpenCL::cl2hpp OR
      NOT TARGET cl2hpp)
    add_library(cl2hpp IMPORTED INTERFACE GLOBAL)
    add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)

    set_target_properties(cl2hpp OpenCL::cl2hpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${download_dir}/..)
  endif()
endif()
