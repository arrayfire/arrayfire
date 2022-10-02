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
  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/mkl/lib")
endif()

if(UNIX AND NOT APPLE)
  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/mkl/lib/intel64")
endif()

