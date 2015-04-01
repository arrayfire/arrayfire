CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_GTEST "Download gtest and check for updates. Necessary if you change compilers" ON)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

OPTION(BUILD_DOCS "Create ArrayFire Documentation" OFF)
OPTION(WITH_COVERAGE "Added code coverage flags" OFF)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
INCLUDE(${CMAKE_MODULE_PATH}/UploadCoveralls.cmake)

FIND_PACKAGE(FreeImage)
IF(FREEIMAGE_FOUND)
    ADD_DEFINITIONS(-DWITH_FREEIMAGE)
    SET(lib_deps ${FREEIMAGE_LIBRARY})
    INCLUDE_DIRECTORIES(${FREEIMAGE_INCLUDE_DIRS})
ELSE(FREEIMAGE_FOUND)
    MESSAGE(WARNING, "FreeImage not found!")
ENDIF(FREEIMAGE_FOUND)

INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/c"
    )

IF(${UNIX})
    ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)
    IF(${WITH_COVERAGE})
        SET(CMAKE_CXX_FLAGS             "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_EXE_LINKER_FLAGS      "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_SHARED_LINKER_FLAGS   "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_STATIC_LINKER_FLAGS   "-fprofile-arcs -ftest-coverage")
    ENDIF(${WITH_COVERAGE})
ENDIF(${UNIX})

# OS Definitions
IF(UNIX)
  IF(APPLE)   #OSX
    ADD_DEFINITIONS(-DOS_MAC)

    SET(CMAKE_MACOSX_RPATH ON)
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    IF("${isSystemDir}" STREQUAL "-1")
      SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF("${isSystemDir}" STREQUAL "-1")
  ELSE(APPLE) #Linux
    ADD_DEFINITIONS(-DOS_LNX)
  ENDIF()
ELSE(${UNIX}) #Windows
  ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF()

# Architechture Definitions
INCLUDE(${CMAKE_MODULE_PATH}/TargetArch.cmake)
target_architecture(ARCH)
IF(${ARCH} STREQUAL "x86_64")
  ADD_DEFINITIONS(-DARCH_64)
ELSE(${ARCH})
  ADD_DEFINITIONS(-DARCH_32)
ENDIF()

INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

IF(WIN32)
    SET(ARRAYFIRE_CMAKE_DIR cmake)
ELSE()
    SET(ARRAYFIRE_CMAKE_DIR share/ArrayFire)
ENDIF()

IF(${BUILD_CPU})
  ADD_SUBDIRECTORY(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/backend/opencl)
ENDIF()

IF(${BUILD_DOCS})
    ADD_SUBDIRECTORY(docs)
ENDIF()

ADD_EXECUTABLE(bin2cpp ${CMAKE_MODULE_PATH}/bin2cpp.cpp)

IF(${BUILD_TEST})
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF()

IF(${BUILD_EXAMPLES})
    ADD_SUBDIRECTORY(examples)
ENDIF()

## Install step
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include
    COMPONENT headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)

## The ArrayFire version file is generated and won't be included above, install
## it separately.
INSTALL(FILES
    ${CMAKE_SOURCE_DIR}/include/af/version.h DESTINATION include/af/
    COMPONENT headers
)

## configuration to be used from the binary directory directly
SET(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(BACKEND_DIR "src/backend/\${lowerbackend}")
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfig.cmake
    @ONLY)

## installed cmake configuration
# use a relative dir to keep arrayfire relocatable
STRING(REGEX REPLACE "[^/]+" ".." reldir "${ARRAYFIRE_CMAKE_DIR}")
SET(INCLUDE_DIR "\${CMAKE_CURRENT_LIST_DIR}/${reldir}/include")
set(BACKEND_DIR)
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    @ONLY)
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfigVersion.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    DESTINATION ${ARRAYFIRE_CMAKE_DIR}
    COMPONENT cmake)

# CPack package generation
# General settings
SET(CPACK_GENERATOR "TGZ")
SET(CPACK_PACKAGE_NAME "arrayfire")
SET(CPACK_PACKAGE_VERSION 3.0.beta)
SET(CPACK_PACKAGE_VERSION_MAJOR "3")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "beta")
SET(CPACK_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
SET(CPACK_PACKAGE_CONTACT "ArrayFire Development Group <technical@arrayfire.com>")

# Long description of the package
SET(CPACK_PACKAGE_DESCRIPTION
"ArrayFire is a high performance software library for parallel computing
with an easy-to-use API. Its array based function set makes parallel
programming simple.

ArrayFire's multiple backends (CUDA, OpenCL and native CPU) make it
platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel
computing code, saving you valuable time and lowering development costs.")

# Short description of the package
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A high performance library for parallel
computing with an easy-to-use API.")

# Useful descriptions for components
SET(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "ArrayFire libraries")
SET(CPACK_COMPONENT_DOCUMENTATION_NAME "Doxygen documentation")
SET(CPACK_COMPONENT_HEADERS_NAME "C/C++ headers")
SET(CPACK_COMPONENT_CMAKE_NAME "CMake support")
# Set the default components installed in the package
SET(CPACK_COMPONENTS_ALL libraries headers documentation cmake)

# Source package
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}_src_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
SET(CPACK_SOURCE_IGNORE_FILES
    "/build"
    "CMakeFiles"
    "/\\\\.dir"
    "/\\\\.git"
    "/\\\\.gitignore$"
    ".*~$"
    "\\\\.bak$"
    "\\\\.swp$"
    "\\\\.orig$"
    "/\\\\.DS_Store$"
    "/Thumbs\\\\.db"
    "/CMakeLists.txt.user$"
    ${CPACK_SOURCE_IGNORE_FILES})
# Ignore build directories that may be in the source tree
FILE(GLOB_RECURSE CACHES "${CMAKE_SOURCE_DIR}/CMakeCache.txt")

# Call to CPACK
INCLUDE(CPack)
