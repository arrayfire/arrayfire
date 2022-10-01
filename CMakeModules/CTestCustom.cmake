# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(CTEST_CUSTOM_ERROR_POST_CONTEXT 200)
set(CTEST_CUSTOM_ERROR_PRE_CONTEXT 200)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS 300)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 300)

if(WIN32)
  if(CMAKE_GENERATOR MATCHES "Ninja")
    set(CTEST_CUSTOM_POST_TEST ./bin/print_info.exe)
  endif()
else()
  set(CTEST_CUSTOM_POST_TEST ./test/print_info)
endif()

list(APPEND CTEST_CUSTOM_COVERAGE_EXCLUDE
  "test"

  # All external and third_party libraries
  "extern/.*"
  "test/mmio/.*"
  "src/backend/cpu/threads/.*"
  "src/backend/cuda/cub/.*"
  "cl2.hpp"

  # Remove bin2cpp from coverage
  "CMakeModules/.*")
