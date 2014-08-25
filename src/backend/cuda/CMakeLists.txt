FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})

ADD_DEFINITIONS(-DAF_CUDA)
REMOVE_DEFINITIONS(-std=c++0x)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/backend/cuda
    )

FILE(GLOB cuda_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "*.cu"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    "kernel/*.cu"
    "kernel/*.cpp"
    "kernel/*.hpp"
    "kernel/*.h"
    )

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "${APPLE}")
    ADD_DEFINITIONS(-stdlib=libstdc++)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang++")
ENDIF()

ADD_DEFINITIONS(-fPIC)

CUDA_ADD_LIBRARY(afcuda SHARED ${cuda_src} OPTIONS "-arch=sm_20")
TARGET_LINK_LIBRARIES(afcuda ${CUDA_LIBRARIES} )
