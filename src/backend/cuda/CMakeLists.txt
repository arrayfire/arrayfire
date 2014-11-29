CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

INCLUDE("${CMAKE_SOURCE_DIR}/common/FindGLEWmx.cmake")
FIND_PACKAGE(GLFW)

IF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)
  SET(GRAPHICS_FOUND ON)
  ADD_DEFINITIONS(-DGLEW_MX -DWITH_GRAPHICS)
ENDIF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)

INCLUDE("${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake")
INCLUDE("${CMAKE_SOURCE_DIR}/common/FindNVVM.cmake")

# Disables running cuda_compute_check.c when build windows using remote
IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  INCLUDE("${CMAKE_SOURCE_DIR}/common/CUDACheckCompute.cmake")
ELSE(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  IF(NOT DEFINED CUDA_GENERATE_CODE)
    SET(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}")
  ENDIF(NOT DEFINED CUDA_GENERATE_CODE)

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
    "${CMAKE_SOURCE_DIR}/src/backend/cuda"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CUDA_NVVM_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    )

FILE(GLOB cuda_headers
     "*.hpp"
     "*.h")

FILE(GLOB cuda_sources
    "*.cu"
    "*.cpp")

FILE(GLOB jit_sources
    "jit/*.hpp")

FILE(GLOB kernel_headers
    "kernel/*.hpp")

source_group(backend\\cuda\\Headers FILES ${cuda_headers})
source_group(backend\\cuda\\Sources FILES ${cuda_sources})
source_group(backend\\cuda\\JIT FILES ${jit_sources})
source_group(backend\\cuda\\kernel\\Headers FILES ${kernel_headers})

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )

FILE(GLOB backend_sources
    "../*.cpp"
    )

source_group(backend\\Headers FILES ${backend_headers})
source_group(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

source_group(api\\c\\Headers FILES ${c_headers})
source_group(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

source_group(api\\cpp\\Sources FILES ${cpp_sources})

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

CUDA_ADD_LIBRARY(afcuda SHARED
                ${cuda_headers}
                ${cuda_sources}
                ${jit_sources}
                ${kernel_headers}
                ${backend_headers}
                ${backend_sources}
                ${c_headers}
                ${c_sources}
                ${cpp_sources}
                OPTIONS "-gencode" ${CUDA_GENERATE_CODE})

ADD_DEPENDENCIES(afcuda ${ptx_targets})


TARGET_LINK_LIBRARIES(afcuda
                     ${CUDA_CUBLAS_LIBRARIES}
                     ${CUDA_LIBRARIES}
                     ${lib_deps}
                     ${CUDA_CUFFT_LIBRARIES}
                     ${CUDA_NVVM_LIBRARIES}
                     ${CUDA_CUDA_LIBRARY})

IF(${GRAPHICS_FOUND})
  TARGET_LINK_LIBRARIES(afcuda
                       ${GLFW_LIBRARY}
                       ${GLEWmx_LIBRARY}
                       ${OPENGL_gl_LIBRARY}
                       ${OPENGL_glu_LIBRARY})
ENDIF()

INSTALL(TARGETS afcuda DESTINATION lib)
