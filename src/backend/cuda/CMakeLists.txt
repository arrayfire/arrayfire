CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

INCLUDE("${CMAKE_MODULE_PATH}/CLKernelToH.cmake")
INCLUDE("${CMAKE_MODULE_PATH}/FindNVVM.cmake")

# Disables running cuda_compute_check.c when build windows using remote
OPTION(CUDA_COMPUTE_DETECT "Run autodetection of CUDA Architecture" ON)
IF(CUDA_COMPUTE_DETECT AND NOT DEFINED CUDA_DETECTED_COMPUTE)
    INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
    SET(CUDA_COMPUTE_${CUDA_DETECTED_COMPUTE} ON CACHE BOOL "" FORCE)
ENDIF()

OPTION(CUDA_COMPUTE_20 "CUDA Compute Capability 2.0" OFF)
OPTION(CUDA_COMPUTE_30 "CUDA Compute Capability 3.0" OFF)
OPTION(CUDA_COMPUTE_32 "CUDA Compute Capability 3.2" OFF)
OPTION(CUDA_COMPUTE_35 "CUDA Compute Capability 3.5" OFF)
OPTION(CUDA_COMPUTE_50 "CUDA Compute Capability 5.0" OFF)
OPTION(CUDA_COMPUTE_52 "CUDA Compute Capability 5.2" OFF)

MARK_AS_ADVANCED(CUDA_COMPUTE_DETECT CUDA_COMPUTE_20
                 CUDA_COMPUTE_30 CUDA_COMPUTE_32 CUDA_COMPUTE_35
                 CUDA_COMPUTE_50 CUDA_COMPUTE_52)

IF(${CUDA_COMPUTE_20})
    SET(VERSION 20)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
ENDIF()
IF(${CUDA_COMPUTE_30})
    SET (VERSION 30)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
ENDIF()
IF(${CUDA_COMPUTE_32})
    SET(VERSION 32)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
ENDIF()
IF(${CUDA_COMPUTE_35})
    SET(VERSION 35)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
ENDIF()
IF(${CUDA_COMPUTE_50})
    SET(VERSION 50)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
ENDIF()
IF(${CUDA_COMPUTE_52})
    SET(VERSION 52)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
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
        FIND_LIBRARY(
            CUDA_cusolver_LIBRARY
            NAMES "cusolver"
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES "lib64" "lib/x64" "lib"
            DOC "CUDA cusolver Library"
            NO_DEFAULT_PATH
            )
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

SOURCE_GROUP(backend\\cuda\\Headers FILES ${cuda_headers})
SOURCE_GROUP(backend\\cuda\\Sources FILES ${cuda_sources})
SOURCE_GROUP(backend\\cuda\\JIT FILES ${jit_sources})
SOURCE_GROUP(backend\\cuda\\kernel\\Headers FILES ${kernel_headers})

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )

FILE(GLOB backend_sources
    "../*.cpp"
    )

SOURCE_GROUP(backend\\Headers FILES ${backend_headers})
SOURCE_GROUP(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

SOURCE_GROUP(api\\c\\Headers FILES ${c_headers})
SOURCE_GROUP(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

SOURCE_GROUP(api\\cpp\\Sources FILES ${cpp_sources})

LIST(LENGTH COMPUTE_VERSIONS COMPUTE_COUNT)
IF(${COMPUTE_COUNT} EQUAL 1)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_GENERATE_CODE}")
ELSE()
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_20")
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
                OPTIONS ${CUDA_GENERATE_CODE})

ADD_DEPENDENCIES(afcuda ${ptx_targets})

TARGET_LINK_LIBRARIES(afcuda
                      ${CUDA_CUBLAS_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${CUDA_cusolver_LIBRARY}
                      ${FreeImage_LIBS}
                      ${CUDA_CUFFT_LIBRARIES}
                      ${CUDA_NVVM_LIBRARIES}
                      ${CUDA_CUDA_LIBRARY})

IF(FORGE_FOUND)
    TARGET_LINK_LIBRARIES(afcuda
                          ${FORGE_LIBRARIES}
                         )
ENDIF()

SET_TARGET_PROPERTIES(afcuda PROPERTIES
    VERSION "${AF_VERSION}"
    SOVERSION "${AF_VERSION_MAJOR}")

IF(INSTALLER_MODE)
    SET_TARGET_PROPERTIES(afcpu PROPERTIES LINK_INTERFACE_LIBRARIES "")
    SET_TARGET_PROPERTIES(afcpu PROPERTIES INTERFACE_LINK_LIBRARIES "")
ENDIF()

INSTALL(TARGETS afcuda EXPORT CUDA DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

export(TARGETS afcuda FILE ArrayFireCUDA.cmake)
INSTALL(EXPORT CUDA DESTINATION "${AF_INSTALL_CMAKE_DIR}"
    COMPONENT cmake
    FILE ArrayFireCUDA.cmake)
