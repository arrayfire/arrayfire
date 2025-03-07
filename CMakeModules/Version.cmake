# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
#
# Make a version file that includes the ArrayFire version and git revision
#
set(AF_VERSION_MAJOR ${ArrayFire_VERSION_MAJOR})
set(AF_VERSION_MINOR ${ArrayFire_VERSION_MINOR})
set(AF_VERSION_PATCH ${ArrayFire_VERSION_PATCH})

set(AF_VERSION ${ArrayFire_VERSION})
set(ArrayFire_API_VERSION_CURRENT ${ArrayFire_VERSION_MAJOR}${ArrayFire_VERSION_MINOR})

# From CMake 3.0.0 CMAKE_<LANG>_COMPILER_ID is AppleClang for OSX machines
# that use clang for compilations
if("${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
    set(COMPILER_NAME "AppleClang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(COMPILER_NAME "LLVM Clang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(COMPILER_NAME "GNU Compiler Collection(GCC/G++)")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(COMPILER_NAME "Intel Compiler")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(COMPILER_NAME "Microsoft Visual Studio")
endif()

set(COMPILER_VERSION "${CMAKE_C_COMPILER_VERSION}")
set(AF_COMPILER_STRING "${COMPILER_NAME} ${COMPILER_VERSION}")

execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT GIT_COMMIT_HASH)
    message(STATUS "No git. Setting hash to default")
    set(GIT_COMMIT_HASH "default")
endif()

configure_file(
    ${ArrayFire_SOURCE_DIR}/CMakeModules/version.h.in
    ${ArrayFire_BINARY_DIR}/include/af/version.h
)

configure_file(
    ${ArrayFire_SOURCE_DIR}/CMakeModules/build_version.hpp.in
    ${ArrayFire_BINARY_DIR}/src/backend/build_version.hpp
)
