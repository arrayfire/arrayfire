

set(CTEST_CUSTOM_ERROR_POST_CONTEXT 30)
set(CTEST_CUSTOM_ERROR_PRE_CONTEXT 30)
set(CTEST_CUSTOM_POST_TEST ./test/print_info)

list(APPEND CTEST_CUSTOM_COVERAGE_EXCLUDE
  "test/gtest/*"

  # All external and third_party libraries
  "extern/*"
  "test/mmio/*"
  "src/backend/cpu/threads/*"
  "src/backend/cuda/cub/*"
  "cl2.hpp"

  # Remove bin2cpp from coverage
  "CMakeModules/*")
