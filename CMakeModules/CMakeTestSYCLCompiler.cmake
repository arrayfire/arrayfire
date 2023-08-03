# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.


if(CMAKE_SYCL_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_SYCL_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# work around enforced code signing and / or missing executable target type
set(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
if(_CMAKE_FEATURE_DETECTION_TARGET_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_CMAKE_FEATURE_DETECTION_TARGET_TYPE})
endif()

# Remove any cached result from an older CMake version.
# We now store this in CMakeSYCLCompiler.cmake.
unset(CMAKE_SYCL_COMPILER_WORKS CACHE)

# Try to identify the ABI and configure it into CMakeSYCLCompiler.cmake
include(CMakeDetermineCompilerABI)
CMAKE_DETERMINE_COMPILER_ABI(SYCL ${ArrayFire_SOURCE_DIR}/CMakeModules/CMakeSYCLCompilerABI.cpp)
if(CMAKE_SYCL_ABI_COMPILED)
  # The compiler worked so skip dedicated test below.
  set(CMAKE_SYCL_COMPILER_WORKS TRUE)
  message(STATUS "Check for working SYCL compiler: ${CMAKE_SYCL_COMPILER} - skipped")
endif()

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected C++ compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_SYCL_COMPILER_WORKS)
  PrintTestCompilerStatus("SYCL")
  __TestCompiler_setTryCompileTargetType()
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testSYCLCompiler.cxx
    "#ifndef __cplusplus\n"
    "# error \"The CMAKE_SYCL_COMPILER is set to a C compiler\"\n"
    "#endif\n"
    "int main(){return 0;}\n")
  # Clear result from normal variable.
  unset(CMAKE_SYCL_COMPILER_WORKS)
  # Puts test result in cache variable.
  try_compile(CMAKE_SYCL_COMPILER_WORKS ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testSYCLCompiler.cxx
    OUTPUT_VARIABLE __CMAKE_SYCL_COMPILER_OUTPUT)
  unset(__TestCompiler_testSYCLCompilerSource)
  # Move result from cache to normal variable.
  set(CMAKE_SYCL_COMPILER_WORKS ${CMAKE_SYCL_COMPILER_WORKS})
  unset(CMAKE_SYCL_COMPILER_WORKS CACHE)
  __TestCompiler_restoreTryCompileTargetType()
  if(NOT CMAKE_SYCL_COMPILER_WORKS)
    PrintTestCompilerResult(CHECK_FAIL "broken")
    string(REPLACE "\n" "\n  " _output "${__CMAKE_SYCL_COMPILER_OUTPUT}")
    message(FATAL_ERROR "The C++ compiler\n  \"${CMAKE_SYCL_COMPILER}\"\n"
      "is not able to compile a simple test program.\nIt fails "
      "with the following output:\n  ${_output}\n\n"
      "CMake will not be able to correctly generate this project.")
  endif()
  PrintTestCompilerResult(CHECK_PASS "works")
endif()

# Try to identify the compiler features
include(CMakeDetermineCompileFeatures)
CMAKE_DETERMINE_COMPILE_FEATURES(SYCL)

set(CMAKE_TRY_COMPILE_CONFIGURATION "")
# Re-configure to save learned information.
configure_file(
  ${ArrayFire_SOURCE_DIR}/CMakeModules/CMakeSYCLCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake
  @ONLY
)
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake)

if(CMAKE_SYCL_SIZEOF_DATA_PTR)
  foreach(f ${CMAKE_SYCL_ABI_FILES})
    include(${f})
  endforeach()
  unset(CMAKE_SYCL_ABI_FILES)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE})
unset(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE)
unset(__CMAKE_SYCL_COMPILER_OUTPUT)
