CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_GTEST "Download gtest and check for updates. Necessary if you change compilers" ON)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/common)

SET(CMAKE_CXX_FLAGS "-Wall -std=c++0x -fvisibility=hidden")

FIND_PACKAGE(FreeImage)
IF(FREEIMAGE_FOUND)
    SET(lib_deps ${FREEIMAGE_LIBRARY} CACHE PATH "Library Dependencies")
ELSE(FREEIMAGE_FOUND)
    MESSAGE(WARNING, "FreeImage not found!")
ENDIF(FREEIMAGE_FOUND)


INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${FREEIMAGE_INCLUDE_PATH}
    )

IF(${UNIX})
    ADD_DEFINITIONS(-Wall -std=c++0x -fvisibility=hidden)
ENDIF()

IF(${BUILD_CPU})
  ADD_SUBDIRECTORY(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    FIND_PACKAGE(CUDA)
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    FIND_PACKAGE(OpenCL)
    ADD_SUBDIRECTORY(src/backend/opencl)
    ADD_EXECUTABLE(bin2cpp common/bin2cpp.cpp)
ENDIF()

ADD_SUBDIRECTORY(src/array)

IF(${BUILD_TEST})
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF()
