# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(ENV{VCPKG_FEATURE_FLAGS} "versions")
set(ENV{VCPKG_KEEP_ENV_VARS} "MKLROOT")
set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)

set(VCPKG_OVERLAY_TRIPLETS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/vcpkg/vcpkg-triplets)
set(VCPKG_OVERLAY_PORTS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/vcpkg/ports)

if(AF_BUILD_CUDA)
  list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
endif()

if(AF_BUILD_OPENCL)
  list(APPEND VCPKG_MANIFEST_FEATURES "opencl")
endif()

if(AF_BUILD_FORGE)
  list(APPEND VCPKG_MANIFEST_FEATURES "forge")
endif()

if(BUILD_TESTING)
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

if(AF_COMPUTE_LIBRARY STREQUAL "Intel-MKL")
  list(APPEND VCPKG_MANIFEST_FEATURES "mkl")
else()
  list(APPEND VCPKG_MANIFEST_FEATURES "openblasfftw")
endif()

if(DEFINED VCPKG_ROOT AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
elseif(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()
