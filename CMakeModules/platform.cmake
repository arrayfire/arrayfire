# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

# Platform specific settings
#
# Add paths and flags specific platforms. This can inc

if(APPLE)
  # Some homebrew libraries(glbinding) are not installed in directories that
  # CMake searches by default.
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/local/opt")

  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/mkl/lib")
endif()

if(UNIX AND NOT APPLE)
  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/mkl/lib/intel64")
endif()

if(WIN32)
  # C4251: Warnings about dll interfaces. Thrown by glbinding, may be fixed in
  #        the future
  # C4068: Warnings about unknown pragmas
  # C4275: Warnings about using non-exported classes as base class of an
  #        exported class
  add_compile_options(/wd4251 /wd4068 /wd4275)

  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};"
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl;"
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64")
endif()
