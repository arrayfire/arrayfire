CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

INCLUDE(${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/common/FindNVVM.cmake)

# Disables running cuda_compute_check.c when build windows using remote
IF(NOT WINDOWS_REMOTE)
  INCLUDE(${CMAKE_SOURCE_DIR}/common/CUDACheckCompute.cmake)
ELSE(NOT WINDOWS_REMOTE)
  IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    MESSAGE(WARNING "CUDA_COMPUTE_CAPABILITY not defined. Fall back to compute 20")
    SET(CUDA_COMPUTE_CAPABILITY "20")
  ENDIF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)

  IF(NOT DEFINED CUDA_GENERATE_CODE)
    SET(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}")
  ENDIF()

  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch compute_${CUDA_COMPUTE_CAPABILITY})
ENDIF()

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_CUDA)
REMOVE_DEFINITIONS(-std=c++0x)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
  ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/backend/cuda
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_NVVM_INCLUDE_DIR}
    )

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )
FILE(GLOB backend_sources
    "../*.cpp"
    )
FILE(GLOB cpp_sources
    "../../array/*.cpp"
    )
source_group(Headers FILES ${backend_headers})
source_group(Sources FILES ${backend_sources})
source_group(CPP\ Sources FILES ${cpp_sources})

FILE(GLOB cuda_headers
  "*.hpp"
    "*.h")
FILE(GLOB cuda_sources
    "*.cu"
  "*.cpp")
source_group(cuda\\Headers FILES ${cuda_headers})
source_group(cuda\\Sources FILES ${cuda_sources})

FILE(GLOB jit_sources
    "jit/*.hpp")
source_group(cuda\\JIT FILES ${jit_sources})

FILE(GLOB kernel_headers
    "kernel/*.hpp")
source_group(cuda\\kernel\\Headers FILES ${kernel_headers})

IF(${CUDA_COMPUTE_CAPABILITY} STREQUAL "21")
    SET(PTX_COMPUTE "20")
ELSEIF(${CUDA_COMPUTE_CAPABILITY} STREQUAL "32")
    SET(PTX_COMPUTE "30")
ELSEIF(${CUDA_COMPUTE_CAPABILITY} STREQUAL "52")
    SET(PTX_COMPUTE "50")
ELSE()
  SET(PTX_COMPUTE ${CUDA_COMPUTE_CAPABILITY})
ENDIF()


FILE(GLOB cuda_ptx "ptx/PTX64/sm_${PTX_COMPUTE}/*.ptx")

SET( ptx_headers
    "ptx_headers")

CL_KERNEL_TO_H(
    SOURCES ${cuda_ptx}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${ptx_headers}
    TARGETS ptx_targets
    NAMESPACE "cuda"
    EOF "1"
    )

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "${APPLE}")
    ADD_DEFINITIONS(-stdlib=libstdc++)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang++")
ENDIF()

CUDA_ADD_LIBRARY(afcuda SHARED ${cpp_sources}
                ${backend_headers}
                ${backend_sources}
                ${cuda_headers}
                ${cuda_sources}
                ${jit_sources}
                ${kernel_headers}
                OPTIONS "-gencode "${CUDA_GENERATE_CODE})
ADD_DEPENDENCIES(afcuda ${ptx_targets})


TARGET_LINK_LIBRARIES(afcuda ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} ${lib_deps} ${CUDA_CUFFT_LIBRARIES} ${CUDA_NVVM_LIBRARIES} ${CUDA_CUDA_LIBRARY})

INSTALL(TARGETS afcuda DESTINATION lib)
