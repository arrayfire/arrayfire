CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_GTEST "Download gtest and check for updates. Necessary if you change compilers" ON)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/common)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    )

SET(CMAKE_CXX_FLAGS "-Wall -std=c++0x -fvisibility=hidden")


IF(${BUILD_CPU})
    SUBDIRS(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    FIND_PACKAGE(CUDA)
    SUBDIRS(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    FIND_PACKAGE(OpenCL)
    SUBDIRS(src/backend/opencl)
    ADD_EXECUTABLE(bin2cpp common/bin2cpp.cpp)
ENDIF()

SUBDIRS(src/array)

IF(${BUILD_TEST})
    ENABLE_TESTING()
    SUBDIRS(test)
ENDIF()
