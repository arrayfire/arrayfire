# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire
  VERSION 3.7.0
  LANGUAGES C CXX )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AFInstallDirs)
include(CMakeDependentOption)
include(InternalUtils)
include(Version)
include(build_cl2hpp)
include(platform)
include(GetPrerequisites)
include(CheckCXXCompilerFlag)

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
find_package(MKL)

# Graphics dependencies
find_package(glbinding QUIET)
find_package(Forge QUIET)
include(boost_package)

option(AF_BUILD_CPU      "Build ArrayFire with a CPU backend"        ON)
option(AF_BUILD_CUDA     "Build ArrayFire with a CUDA backend"       ${CUDA_FOUND})
option(AF_BUILD_OPENCL   "Build ArrayFire with a OpenCL backend"     ${OpenCL_FOUND})
option(AF_BUILD_UNIFIED  "Build Backend-Independent ArrayFire API"   ON)
option(AF_BUILD_DOCS     "Create ArrayFire Documentation"            ${DOXYGEN_FOUND})
option(AF_BUILD_EXAMPLES "Build Examples"                            ON)

option(AF_WITH_GRAPHICS "Build ArrayFire with Forge Graphics"  $<AND:${OPENGL_FOUND},${Forge_FOUND},${glbinding_FOUND}>)
option(AF_WITH_NONFREE  "Build ArrayFire nonfree algorithms"   OFF)

option(AF_INSTALL_STANDALONE "Build installers that include all dependencies" OFF)

cmake_dependent_option(AF_WITH_RELATIVE_TEST_DIR "Use relative paths for the test data directory(For continious integration(CI) purposes only)" OFF
  "BUILD_TESTING" OFF)

cmake_dependent_option(AF_USE_SYSTEM_FORGE "Use system Forge" OFF
                       "AF_WITH_GRAPHICS" OFF)
cmake_dependent_option(AF_WITH_IMAGEIO "Build ArrayFire with Image IO support" ${FreeImage_FOUND}
                       "FreeImage_FOUND" OFF)
cmake_dependent_option(AF_BUILD_FRAMEWORK "Build an ArrayFire framework for Apple platforms.(Experimental)" OFF
                       "APPLE" OFF)
option(AF_WITH_STATIC_FREEIMAGE "Use Static FreeImage Lib" OFF)

set(AF_WITH_CPUID ON CACHE BOOL "Build with CPUID integration")

af_deprecate(BUILD_CPU             AF_BUILD_CPU)
af_deprecate(BUILD_CUDA            AF_BUILD_CUDA)
af_deprecate(BUILD_OPENCL          AF_BUILD_OPENCL)
af_deprecate(BUILD_UNIFIED         AF_BUILD_UNIFIED)
af_deprecate(BUILD_GRAPHICS        AF_WITH_GRAPHICS)
af_deprecate(BUILD_DOCS            AF_BUILD_DOCS)
af_deprecate(BUILD_NONFREE         AF_WITH_NONFREE)
af_deprecate(BUILD_EXAMPLES        AF_BUILD_EXAMPLES)
af_deprecate(USE_RELATIVE_TEST_DIR AF_WITH_RELATIVE_TEST_DIR)
af_deprecate(USE_FREEIMAGE_STATIC  AF_WITH_STATIC_FREEIMAGE)
af_deprecate(USE_CPUID             AF_WITH_CPUID)

mark_as_advanced(
  AF_BUILD_FRAMEWORK
  AF_INSTALL_STANDALONE
  AF_USE_SYSTEM_FORGE
  AF_WITH_CPUID
  CUDA_HOST_COMPILER
  CUDA_USE_STATIC_CUDA_RUNTIME
  CUDA_rt_LIBRARY)

# TODO(umar): Add definitions should not be used. Instead use
arrayfire_get_platform_definitions(platform_definitions)
add_definitions(${platform_definitions})

if(AF_WITH_GRAPHICS AND NOT AF_USE_SYSTEM_FORGE)
  include(build_forge)
endif()

configure_file(
    ${ArrayFire_SOURCE_DIR}/CMakeModules/version.hpp.in
    ${ArrayFire_BINARY_DIR}/version.hpp
)

if(AF_WITH_NONFREE)
  message("Building with NONFREE requires the following patents")
  message("Method and apparatus for identifying scale invariant features\n"
    "in an image and use of same for locating an object in an image, David\n"
    "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application\n"
    "filed March 8, 1999. Asignee: The University of British Columbia. For\n"
    "further details, contact David Lowe (lowe@cs.ubc.ca) or the\n"
    "University-Industry Liaison Office of the University of British\n"
    "Columbia.")
endif()

# when crosscompiling use the bin2cpp file from the native bin directory
if(CMAKE_CROSSCOMPILING)
  set(NATIVE_BIN_DIR "NATIVE_BIN_DIR-NOTFOUND"
    CACHE FILEPATH "Path to the Native build directory.")
  if(NATIVE_BIN_DIR)
    include(${NATIVE_BIN_DIR}/ImportExecutables.cmake)
  else()
    message(SEND_ERROR "Native Directory not found. Run cmake in a separate"
                       "directory and build the bin2cpp target.")
  endif()
else()
  add_executable(bin2cpp ${ArrayFire_SOURCE_DIR}/CMakeModules/bin2cpp.cpp)
  target_link_libraries(bin2cpp)
  export(TARGETS bin2cpp FILE ${CMAKE_BINARY_DIR}/ImportExecutables.cmake)
endif()

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

conditional_directory(AF_BUILD_CPU     src/backend/cpu)
conditional_directory(AF_BUILD_CUDA    src/backend/cuda)
conditional_directory(AF_BUILD_OPENCL  src/backend/opencl)
conditional_directory(AF_BUILD_UNIFIED src/api/unified)

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

# On some distributions the linker will not add a library to the ELF header if
# the symbols are not needed when the library was first parsed by the linker.
# This causes undefined references issues when linking with libraries which have
# circular dependencies.
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set_target_properties(${built_backends} PROPERTIES
                        LINK_FLAGS "-Wl,--no-as-needed")
endif()

foreach(backend ${built_backends})
  target_compile_definitions(${backend} PRIVATE AFDLL)
endforeach()

if(AF_BUILD_FRAMEWORK)
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
install(FILES ${ArrayFire_BINARY_DIR}/include/af/version.h
        DESTINATION "${AF_INSTALL_INC_DIR}/af/"
        COMPONENT headers)

if(Forge_FOUND AND NOT AF_USE_SYSTEM_FORGE)
    option(AF_INSTALL_FORGE_DEV "Install Forge Header and Share Files with ArrayFire" OFF)
    mark_as_advanced(AF_INSTALL_FORGE_DEV)
    af_deprecate(INSTALL_FORGE_DEV AF_INSTALL_FORGE_DEV)
    if(AF_INSTALL_FORGE_DEV)
        install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/include/"
                DESTINATION "${AF_INSTALL_INC_DIR}"
                COMPONENT headers
        )
        install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/share/Forge/"
                DESTINATION "${AF_INSTALL_DATA_DIR}/../Forge"
                COMPONENT share
        )
        install(DIRECTORY "${ArrayFire_BINARY_DIR}/third_party/forge/lib/"
                DESTINATION "${AF_INSTALL_LIB_DIR}"
                COMPONENT forge
        )
    endif()
    #install forge library & dependencies
    set(fg_dlib_px "bin")
    if (UNIX)
      set(fg_dlib_px "lib")
    endif ()
    install(DIRECTORY "${PROJECT_BINARY_DIR}/third_party/forge/${fg_dlib_px}/"
      DESTINATION "${AF_INSTALL_BIN_DIR}"
      COMPONENT common_backend_dependencies)
endif()

# install the examples irrespective of the AF_BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on AF_BUILD_EXAMPLES
# when AF_BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${AF_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)

install(DIRECTORY assets/examples/ #NOTE The slash at the end is important
    DESTINATION ${AF_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)

install(DIRECTORY "${ArrayFire_SOURCE_DIR}/LICENSES/"
    DESTINATION LICENSES
    COMPONENT licenses)

foreach(backend CPU CUDA OpenCL Unified)
  string(TOUPPER ${backend} upper_backend)
  string(TOLOWER ${backend} lower_backend)
  if(AF_BUILD_${upper_backend})
    install(EXPORT ArrayFire${backend}Targets
            NAMESPACE ArrayFire::
            DESTINATION ${AF_INSTALL_CMAKE_DIR}
            COMPONENT ${lower_backend})

    export( EXPORT ArrayFire${backend}Targets
            NAMESPACE ArrayFire::
            FILE cmake/ArrayFire${backend}Targets.cmake)
  endif()
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${ArrayFire_BINARY_DIR}/ArrayFireConfigVersion.cmake"
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
              ${ArrayFire_BINARY_DIR}/ArrayFireConfigVersion.cmake
              DESTINATION ${AF_INSTALL_CMAKE_DIR}
              COMPONENT cmake)

if((USE_CPU_MKL OR USE_OPENCL_MKL) AND TARGET MKL::MKL AND AF_INSTALL_STANDALONE)
  if(TARGET MKL::ThreadingLibrary)
    install(FILES
      $<TARGET_FILE:MKL::ThreadingLibrary>
      DESTINATION ${AF_INSTALL_BIN_DIR}
      COMPONENT mkl_dependencies)
  endif()

  if(NOT WIN32)
    install(FILES
      $<TARGET_FILE:MKL::Interface>
      DESTINATION ${AF_INSTALL_BIN_DIR}
      COMPONENT mkl_dependencies)
  endif()

  install(FILES
    $<TARGET_FILE:MKL::MKL>
    $<TARGET_FILE:MKL::ThreadLayer>
    ${MKL_RUNTIME_KERNEL_LIBRARIES}

    # This variable is used to add tbb.so.2 library because the main lib
    # is a linker script and not a symlink so it cant be resolved using
    # get_filename_component
    ${AF_ADDITIONAL_MKL_LIBRARIES}
    DESTINATION ${AF_INSTALL_BIN_DIR}
    COMPONENT mkl_dependencies)
endif()

# This file will be used to create the config file for the build directory.
# These config files will be used by the examples to find the ArrayFire
# libraries
set(INCLUDE_DIRS "${ArrayFire_SOURCE_DIR}/include" "${ArrayFire_BINARY_DIR}/include")
set(CMAKE_DIR "${ArrayFire_BINARY_DIR}/cmake")
configure_package_config_file(
  ${CMAKE_MODULE_PATH}/ArrayFireConfig.cmake.in
  ArrayFireConfig.cmake
  INSTALL_DESTINATION "${ArrayFire_BINARY_DIR}"
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  INSTALL_PREFIX "${ArrayFire_BINARY_DIR}"
  )

# Registers the current build directory with the user's cmake config. This will
# create a file at $HOME/.cmake/packages/ArrayFire which will point to this source
# build directory.
# TODO(umar): Disable for now. Causing issues with builds on windows.
#export(PACKAGE ArrayFire)

include(CTest)

# Handle depricated BUILD_TEST variable if found.
if(BUILD_TEST)
  set(BUILD_TESTING ${BUILD_TEST})
endif()

conditional_directory(BUILD_TESTING test)

set(ASSETS_DIR "${ArrayFire_SOURCE_DIR}/assets")
conditional_directory(AF_BUILD_EXAMPLES examples)
conditional_directory(AF_BUILD_DOCS docs)

include(CPackConfig)

