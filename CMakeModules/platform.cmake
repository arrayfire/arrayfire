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
  # IMP NOTE: After removing link time dependency of gfx libs, glbinding is
  #           still needed in cmake's prefix path so that forge doesn't fail
  #           in cmake generation phase because of no glbinding.
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
  # C4068: Warnings about unknown pragmas
  # C4275: Warnings about using non-exported classes as base class of an
  #        exported class
  add_compile_options(/wd4068 /wd4275)
endif()
