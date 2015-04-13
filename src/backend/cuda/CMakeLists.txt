CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

INCLUDE("${CMAKE_MODULE_PATH}/CLKernelToH.cmake")
INCLUDE("${CMAKE_MODULE_PATH}/FindNVVM.cmake")

# Disables running cuda_compute_check.c when build windows using remote
IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
ELSE(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  IF(NOT DEFINED CUDA_GENERATE_CODE)
    SET(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}")
  ENDIF(NOT DEFINED CUDA_GENERATE_CODE)

  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch compute_${CUDA_COMPUTE_CAPABILITY})
ENDIF()

IF(UNIX)
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fvisibility=hidden)
  REMOVE_DEFINITIONS(-std=c++0x)
  IF(${WITH_COVERAGE})
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fprofile-arcs -Xcompiler -ftest-coverage -Xlinker -fprofile-arcs -Xlinker -ftest-coverage")
  ENDIF(${WITH_COVERAGE})
ELSE()
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_CUDA)

IF(${CUDA_VERSION_MAJOR} LESS 7)
    MESSAGE(STATUS "CUDA Version ${CUDA_VERSION_STRING} does not contain cuSolve library. Linear Algebra will not be available." )
    IF(CMAKE_VERSION VERSION_LESS 3.2)
        SET(CUDA_cusolver_LIBRARY)
    ENDIF(CMAKE_VERSION VERSION_LESS 3.2)
ELSE(${CUDA_VERSION_MAJOR} LESS 7)
    MESSAGE(STATUS "CUDA cusolver library available in CUDA Version ${CUDA_VERSION_STRING}")
    ADD_DEFINITIONS(-DWITH_CUDA_LINEAR_ALGEBRA)
    IF(CMAKE_VERSION VERSION_LESS 3.2)
        SET(CUDA_cusolver_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusolver.so)
    ENDIF(CMAKE_VERSION VERSION_LESS 3.2)
ENDIF(${CUDA_VERSION_MAJOR} LESS 7)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/src/backend/cuda"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CUDA_NVVM_INCLUDE_DIR}
    )

FILE(GLOB cuda_headers
     "*.hpp"
     "*.h")

FILE(GLOB cuda_sources
    "*.cu"
    "*.cpp"
    "sort_by_key/*.cu"
    "kernel/*.cu")

FILE(GLOB jit_sources
    "JIT/*.hpp")

FILE(GLOB kernel_headers
    "kernel/*.hpp")

FILE(GLOB ptx_sources
    "JIT/*.cu")

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

CUDA_COMPILE_PTX(ptx_files ${ptx_sources})

set(cuda_ptx "")
foreach(ptx_src_file ${ptx_sources})

      get_filename_component(_name "${ptx_src_file}" NAME_WE)

      set(_gen_file_name
        "${CMAKE_BINARY_DIR}/src/backend/cuda/cuda_compile_ptx_generated_${_name}.cu.ptx")
      set(_out_file_name
        "${CMAKE_BINARY_DIR}/src/backend/cuda/${_name}.ptx")

      ADD_CUSTOM_COMMAND(
        OUTPUT "${_out_file_name}"
        DEPENDS "${_gen_file_name}"
        COMMAND ${CMAKE_COMMAND} -E copy "${_gen_file_name}" "${_out_file_name}")

      list(APPEND cuda_ptx "${_out_file_name}")
endforeach()

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

IF("${APPLE}")
    ADD_DEFINITIONS(-D__STRICT_ANSI__)
    IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        IF(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
            SET(STD_LIB_BINDING "-stdlib=libstdc++")
        ELSE(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
            SET(STD_LIB_BINDING "-stdlib=libc++")
        ENDIF()

        ADD_DEFINITIONS("${STD_LIB_BINDING}")
        SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${STD_LIB_BINDING}")
        SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${STD_LIB_BINDING}")
        SET(CUDA_HOST_COMPILER "/usr/bin/clang++")
    ENDIF()
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
                      ${CUDA_cusolver_LIBRARY}
                      ${FreeImage_LIBS}
                      ${CUDA_CUFFT_LIBRARIES}
                      ${CUDA_NVVM_LIBRARIES}
                      ${CUDA_CUDA_LIBRARY})

SET_TARGET_PROPERTIES(afcuda PROPERTIES
    VERSION "${AF_VERSION}"
    SOVERSION "${AF_VERSION_MAJOR}")

INSTALL(TARGETS afcuda EXPORT CUDA DESTINATION lib COMPONENT libraries)

export(TARGETS afcuda FILE ArrayFireCUDA.cmake)
INSTALL(EXPORT CUDA DESTINATION ${ARRAYFIRE_CMAKE_DIR}
    COMPONENT cmake
    FILE ArrayFireCUDA.cmake)
