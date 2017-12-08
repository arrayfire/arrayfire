# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire
  VERSION 3.6.0
  LANGUAGES C CXX )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AFInstallDirs)
include(CMakeDependentOption)
include(InternalUtils)
include(Version)
include(build_cl2hpp)

arrayfire_set_cmake_default_variables()

find_package(CUDA 7.0)
find_package(OpenCL 1.2)
find_package(OpenGL)
find_package(OpenMP)
find_package(FreeImage)
find_package(Threads)
find_package(FFTW)
find_package(CBLAS)
find_package(LAPACKE)
find_package(Doxygen)

# Graphics dependencies
find_package(glbinding QUIET)
find_package(Boost)

option(BUILD_CPU      "Build ArrayFire with a CPU backend"        ON)
option(BUILD_CUDA     "Build ArrayFire with a CUDA backend"       ${CUDA_FOUND})
option(BUILD_OPENCL   "Build ArrayFire with a OpenCL backend"     ${OpenCL_FOUND})
option(BUILD_UNIFIED  "Build Backend-Independent ArrayFire API"   ON)

option(BUILD_GRAPHICS "Build ArrayFire with Forge Graphics"      $<AND:${OPENGL_FOUND},${Forge_FOUND},${glbinding_FOUND}>)
option(BUILD_DOCS     "Create ArrayFire Documentation"           ${DOXYGEN_FOUND})
option(BUILD_NONFREE  "Build ArrayFire nonfree algorithms"       OFF)

option(BUILD_EXAMPLES "Build Examples"                           ON)
cmake_dependent_option(USE_RELATIVE_TEST_DIR "Use relative paths for the test data directory(For continious integration(CI) purposes only)" OFF
  "BUILD_TESTING" OFF)

cmake_dependent_option(USE_SYSTEM_FORGE "Use system Forge" OFF
                    "BUILD_GRAPHICS" OFF)
cmake_dependent_option(WITH_IMAGEIO "Build ArrayFire with Image IO support" ${FreeImage_FOUND}
                      "FreeImage_FOUND" OFF)
cmake_dependent_option(BUILD_FRAMEWORK "Build an ArrayFire framework for Apple platforms.(Experimental)" OFF
                      "APPLE" OFF)
option(USE_FREEIMAGE_STATIC "Use Static FreeImage Lib" OFF)

set(USE_CPUID ON CACHE BOOL "Build with CPUID integration")

mark_as_advanced(
  BUILD_FRAMEWORK
  USE_SYSTEM_FORGE
  USE_CPUID)

# TODO(umar): Add definitions should not be used. Instead use
arrayfire_get_platform_definitions(platform_definitions)
add_definitions(${platform_definitions})

if(WIN32)
  #TODO(umar): create a single place for compiler specific settings
  # C4251: Warnings about dll interfaces. Thrown by glbinding, may be fixed in
  #        the future
  # C4068: Warnings about unknown pragmas
  # C4275: Warnings about using non-exported classes as base class of an
  #        exported class
  add_compile_options(/wd4251 /wd4068 /wd4275)
endif()

if(BUILD_GRAPHICS)
  include(build_forge)
endif()

configure_file(
    ${ArrayFire_SOURCE_DIR}/CMakeModules/version.hpp.in
    ${ArrayFire_BINARY_DIR}/version.hpp
)

if(BUILD_NONFREE)
  message("Building with NONFREE requires the following patents")
  message("Method and apparatus for identifying scale invariant features\n"
    "in an image and use of same for locating an object in an image, David\n"
    "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application\n"
    "filed March 8, 1999. Asignee: The University of British Columbia. For\n"
    "further details, contact David Lowe (lowe@cs.ubc.ca) or the\n"
    "University-Industry Liaison Office of the University of British\n"
    "Columbia.")
endif()

add_executable(bin2cpp ${ArrayFire_SOURCE_DIR}/CMakeModules/bin2cpp.cpp)

if(NOT LAPACK_FOUND)
    if(APPLE)
        # UNSET THE VARIABLES FROM LAPACKE
        unset(LAPACKE_LIB CACHE)
        unset(LAPACK_LIB CACHE)
        unset(LAPACKE_INCLUDES CACHE)
        unset(LAPACKE_ROOT_DIR CACHE)
        find_package(LAPACK)
    endif()
endif()

add_subdirectory(src/backend/common)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)

conditional_directory(BUILD_CPU     src/backend/cpu)
conditional_directory(BUILD_CUDA    src/backend/cuda)
conditional_directory(BUILD_OPENCL  src/backend/opencl)
conditional_directory(BUILD_UNIFIED src/api/unified)

if(TARGET af)
  list(APPEND built_backends af)
endif()

if(TARGET afcpu)
  list(APPEND built_backends afcpu)
endif()

if(TARGET afcuda)
  list(APPEND built_backends afcuda)
endif()

if(TARGET afopencl)
  list(APPEND built_backends afopencl)
endif()

set_target_properties(${built_backends} PROPERTIES
                      VERSION "${ArrayFire_VERSION}"
                      SOVERSION "${ArrayFire_VERSION_MAJOR}")

foreach(backend ${built_backends})
  target_compile_definitions(${backend} PRIVATE AFDLL)
endforeach()

if(BUILD_FRAMEWORK)
  set_target_properties(${built_backends}
    PROPERTIES
      FRAMEWORK TRUE
      FRAMEWORK_VERSION A
      MACOSX_FRAMEWORK_IDENTIFIER com.arrayfire.arrayfireFramework
      #MACOSX_FRAMEWORK_INFO_PLIST Info.plist
      #PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/arrayfire.h;${af_headers}"
      #XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
    )
endif()

install(DIRECTORY include/ DESTINATION ${AF_INSTALL_INC_DIR}
    COMPONENT headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)

## The ArrayFire version file is generated and won't be included above, install
## it separately.
install(FILES
    ${ArrayFire_BINARY_DIR}/include/af/version.h DESTINATION "${AF_INSTALL_INC_DIR}/af/"
    COMPONENT headers
)

if(Forge_FOUND AND NOT USE_SYSTEM_FORGE)
    option(INSTALL_FORGE_DEV "Install Forge Header and Share Files with ArrayFire" OFF)
    install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/lib/"
            DESTINATION "${AF_INSTALL_LIB_DIR}"
            COMPONENT forge
    )
    if(${INSTALL_FORGE_DEV})
        install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/include/"
                DESTINATION "${AF_INSTALL_INC_DIR}"
                COMPONENT headers
        )
        install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/share/Forge/"
                DESTINATION "${AF_INSTALL_DATA_DIR}/../Forge"
                COMPONENT share
        )
    endif()
endif()

# install the examples irrespective of the BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on BUILD_EXAMPLES
# when BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${AF_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)

install(DIRECTORY assets/examples/ #NOTE The slash at the end is important
    DESTINATION ${AF_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)

foreach(backend CPU CUDA OpenCL Unified)
  string(TOUPPER ${backend} upper_backend)
  if(BUILD_${upper_backend})
    install(EXPORT ArrayFire${backend}Targets
            NAMESPACE ArrayFire::
            DESTINATION ${AF_INSTALL_CMAKE_DIR}
            COMPONENT cmake)

    export( EXPORT ArrayFire${backend}Targets
            NAMESPACE ArrayFire::
            FILE cmake/ArrayFire${backend}Targets.cmake)
  endif()
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${ArrayFire_BINARY_DIR}/cmake/ArrayFireConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

# This config file will be installed so we need to set the install_destination
# path relitive to the install path
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${AF_INSTALL_CMAKE_DIR})
configure_package_config_file(
  ${CMAKE_MODULE_PATH}/ArrayFireConfig.cmake.in
  cmake/install/ArrayFireConfig.cmake
  INSTALL_DESTINATION "${AF_INSTALL_CMAKE_DIR}"
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  )

install(FILES ${ArrayFire_BINARY_DIR}/cmake/install/ArrayFireConfig.cmake
              ${ArrayFire_BINARY_DIR}/cmake/ArrayFireConfigVersion.cmake
              DESTINATION ${AF_INSTALL_CMAKE_DIR}
              COMPONENT cmake)


# This file will be used to create the config file for the build directory.
# These config files will be used by the examples to find the ArrayFire
# libraries
set(INCLUDE_DIRS "${ArrayFire_SOURCE_DIR}/include" "${ArrayFire_BINARY_DIR}/include")
set(CMAKE_DIR "${ArrayFire_BINARY_DIR}/cmake")
configure_package_config_file(
  ${CMAKE_MODULE_PATH}/ArrayFireConfig.cmake.in
  cmake/ArrayFireConfig.cmake
  INSTALL_DESTINATION "${ArrayFire_BINARY_DIR}/cmake"
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  INSTALL_PREFIX "${ArrayFire_BINARY_DIR}"
  )

# Registers the current build directory with the user's cmake config. This will
# create a file at $HOME/.cmake/packages/ArrayFire which will point to this source
# build directory.
# TODO(umar): Disable for now. Causing issues with builds on windows.
#export(PACKAGE ArrayFire)

include(CPackConfig)
include(CTest)

# Handle depricated BUILD_TEST variable if found.
if(BUILD_TEST)
  set(BUILD_TESTING ${BUILD_TEST})
endif()

conditional_directory(BUILD_TESTING test)
conditional_directory(BUILD_EXAMPLES examples)
conditional_directory(BUILD_DOCS docs)
