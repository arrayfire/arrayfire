# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

FetchContent_Declare(
  ${clfft_prefix}
  GIT_REPOSITORY    https://github.com/arrayfire/clFFT.git
  GIT_TAG           cmake_fixes
)
FetchContent_Populate(${clfft_prefix})

set(current_build_type ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${${clfft_prefix}_SOURCE_DIR}/src ${${clfft_prefix}_BINARY_DIR} EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${current_build_type})

mark_as_advanced(
  Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE
  CLFFT_BUILD64
  CLFFT_BUILD_CALLBACK_CLIENT
  CLFFT_BUILD_CLIENT
  CLFFT_BUILD_EXAMPLES
  CLFFT_BUILD_LOADLIBRARIES
  CLFFT_BUILD_RUNTIME
  CLFFT_BUILD_TEST
  CLFFT_CODE_COVERAGE
  CLFFT_SUFFIX_BIN
  CLFFT_SUFFIX_LIB
)
