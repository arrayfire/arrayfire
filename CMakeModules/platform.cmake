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

if(WIN32)
  # C4068: Warnings about unknown pragmas
  # C4275: Warnings about using non-exported classes as base class of an
  #        exported class
  add_compile_options(/wd4068 /wd4275)

  # MSVC incorrectly sets the cplusplus to 199711L even if the compiler supports
  # c++11 features. This flag sets it to the correct standard supported by the
  # compiler
  check_cxx_compiler_flag(/Zc:__cplusplus cplusplus_define)
  if(cplusplus_define)
    add_compile_options(/Zc:__cplusplus)
  endif()

  # The "permissive-" option enforces strict(er?) standards compliance by
  # MSVC
  check_cxx_compiler_flag(/permissive- cxx_compliance)
  if(cxx_compliance)
    add_compile_options(/permissive-)
  endif()
endif()
