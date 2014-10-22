INCLUDE(${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/common/FindNVVM.cmake)

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})

ADD_DEFINITIONS(-DAF_CUDA)
REMOVE_DEFINITIONS(-std=c++0x)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/backend/cuda
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_NVVM_INCLUDE_DIR}
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

FILE(GLOB cuda_ptx "ptx/PTX64/sm_20/*.ptx")

SET( ptx_headers
    "ptx_headers")

CL_KERNEL_TO_H(
    SOURCES ${cuda_ptx}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${ptx_headers}
    TARGETS ptx_targets
    NAMESPACE "cuda"
    )

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "${APPLE}")
    ADD_DEFINITIONS(-stdlib=libstdc++)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang++")
ENDIF()

CUDA_ADD_LIBRARY(afcuda SHARED ${cuda_src} OPTIONS "-arch=sm_20")
ADD_DEPENDENCIES(afcuda ${ptx_targets})


find_library(CUDA_DRIVER_LIBRARY cuda)

TARGET_LINK_LIBRARIES(afcuda ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} ${lib_deps} ${CUDA_CUFFT_LIBRARIES} ${CUDA_NVVM_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
