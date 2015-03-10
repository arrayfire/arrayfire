CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_GTEST "Download gtest and check for updates. Necessary if you change compilers" ON)
OPTION(USE_SYSTEM_GTEST "Use libgtest already present on the system" OFF)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

OPTION(BUILD_DOCS "Create ArrayFire Documentation" OFF)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

FIND_PACKAGE(FreeImage)
IF(FREEIMAGE_FOUND)
  ADD_DEFINITIONS(-DWITH_FREEIMAGE)
    SET(lib_deps ${FREEIMAGE_LIBRARY} CACHE PATH "Library Dependencies")
ELSE(FREEIMAGE_FOUND)
    MESSAGE(WARNING, "FreeImage not found!")
ENDIF(FREEIMAGE_FOUND)


INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/c"
    ${FREEIMAGE_INCLUDE_PATH}
    )

IF(${UNIX})
    ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)
ENDIF()

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
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)

## The ArrayFire version file is generated and won't be included above, install
## it separately.
INSTALL(FILES 
    ${CMAKE_SOURCE_DIR}/include/af/version.h DESTINATION include/af/
)
