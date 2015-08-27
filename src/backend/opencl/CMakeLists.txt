CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)
PROJECT(ARRAYFIRE)

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE("${CMAKE_MODULE_PATH}/CLKernelToH.cmake")

IF(USE_OPENCL_F77_BLAS)
    MESSAGE("Using F77 BLAS")
    ADD_DEFINITIONS(-DUSE_F77_BLAS)
ENDIF()

IF(USE_OPENCL_MKL)
    MESSAGE("Using MKL")
    ADD_DEFINITIONS(-DUSE_MKL)
ENDIF()

IF(APPLE)
    FIND_PACKAGE(LAPACK)
ELSE(APPLE) # Linux and Windows
    FIND_PACKAGE(LAPACKE)
ENDIF(APPLE)

IF(NOT LAPACK_FOUND)
    MESSAGE(WARNING "LAPACK not found. Functionality will be disabled")
ELSE(NOT LAPACK_FOUND)
  ADD_DEFINITIONS(-DWITH_OPENCL_LINEAR_ALGEBRA)

  IF(NOT USE_OPENCL_MKL)
    FIND_PACKAGE(CBLAS REQUIRED)

    IF(USE_CPU_F77_BLAS)
      MESSAGE("Using F77 BLAS")
      ADD_DEFINITIONS(-DUSE_F77_BLAS)
    ENDIF()

    IF (NOT CBLAS_LIBRARIES)
      MESSAGE(SEND_ERROR "CBLAS Library not set")
    ELSE()
      MESSAGE(STATUS "Using CBLAS Library: ${CBLAS_LIBRARIES}")
    ENDIF()
  ENDIF()
ENDIF()

IF(NOT UNIX)
    ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_OPENCL
                -D__CL_ENABLE_EXCEPTIONS)

OPTION(USE_SYSTEM_CLBLAS "Use system clBLAS" OFF)
IF(USE_SYSTEM_CLBLAS)
    FIND_PACKAGE(clBLAS REQUIRED)
ELSE()
    INCLUDE("${CMAKE_MODULE_PATH}/build_clBLAS.cmake")
ENDIF()
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})

OPTION(USE_SYSTEM_CLFFT "Use system clFFT" OFF)
IF(USE_SYSTEM_CLFFT)
    FIND_PACKAGE(clFFT REQUIRED)
ELSE()
    INCLUDE("${CMAKE_MODULE_PATH}/build_clFFT.cmake")
ENDIF()
INCLUDE_DIRECTORIES(${CLFFT_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLFFT_LIBRARY_DIR})

ADD_DEFINITIONS( -DBOOST_ALL_NO_LIB )
SET(Boost_USE_STATIC_LIBS OFF)
FIND_PACKAGE(Boost 1.48 REQUIRED)

OPTION(USE_SYSTEM_BOOST_COMPUTE "Use system BoostCompute" OFF)
IF(USE_SYSTEM_BOOST_COMPUTE)
    FIND_PACKAGE(BoostCompute REQUIRED)
ELSE()
    INCLUDE("${CMAKE_MODULE_PATH}/build_boost_compute.cmake")
ENDIF()

SET( cl_kernel_headers
    "kernel_headers")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend/opencl"
    ${OpenCL_INCLUDE_DIRS}
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CLBLAS_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${BoostCompute_INCLUDE_DIRS}
    ${CBLAS_INCLUDE_DIR}
    ${LAPACK_INCLUDE_DIR}
    )

FILE(GLOB opencl_headers
  "*.hpp"
  "*.h"
  "sort_by_key/*.hpp")

FILE(GLOB opencl_sources
    "*.cpp"
    "sort_by_key/*.cpp")

FILE(GLOB magma_sources
    "magma/*.cpp")

FILE(GLOB magma_headers
    "magma/*.h")

FILE(GLOB jit_sources
    "jit/*.hpp")

FILE(GLOB kernel_headers
    "kernel/*.hpp")

FILE(GLOB opencl_kernels
    "kernel/*.cl")

FILE(GLOB kernel_sources
     "kernel/*.cpp")

FILE(GLOB conv_ker_headers
    "kernel/convolve/*.hpp")

FILE(GLOB conv_ker_sources
     "kernel/convolve/*.cpp")

source_group(backend\\opencl\\Headers FILES ${opencl_headers})
source_group(backend\\opencl\\Sources FILES ${opencl_sources})
source_group(backend\\opencl\\JIT FILES ${jit_sources})
source_group(backend\\opencl\\kernel\\Headers FILES ${kernel_headers})
source_group(backend\\opencl\\kernel\\cl FILES ${opencl_kernels})
source_group(backend\\opencl\\kernel\\Sources FILES ${kernel_sources})
source_group(backend\\opencl\\kernel\\convolve\\Headers FILES ${conv_ker_headers})
source_group(backend\\opencl\\kernel\\convolve\\Sources FILES ${conv_ker_sources})
source_group(backend\\opencl\\magma\\Sources FILES ${magma_sources})
source_group(backend\\opencl\\magma\\Headers FILES ${magma_headers})

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

FILE(GLOB kernel_src ${opencl_kernels} "kernel/KParam.hpp")

CL_KERNEL_TO_H(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${cl_kernel_headers}
    TARGETS cl_kernel_targets
    NAMESPACE "opencl"
    EOF "0"
    )

# OS Definitions
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread -Wno-comment")
ELSE(${UNIX}) #Windows
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
    SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /bigobj")
ENDIF()

IF(DEFINED BLAS_SYM_FILE)

    ADD_LIBRARY(afopencl_static STATIC
                ${opencl_headers}
                ${opencl_sources}
                ${jit_sources}
                ${kernel_headers}
                ${opencl_kernels}
                ${kernel_sources}
                ${conv_ker_headers}
                ${conv_ker_sources}
                ${backend_headers}
                ${backend_sources}
                ${magma_sources}
                ${magma_headers})

    ADD_LIBRARY(afopencl SHARED
               ${c_headers}
               ${c_sources}
               ${cpp_sources})


    IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
        ADD_DEPENDENCIES(afopencl_static forge)
    ENDIF()

    IF(APPLE)
    SET_TARGET_PROPERTIES(afopencl_static
        PROPERTIES LINK_FLAGS -Wl,-exported_symbols_list,${BLAS_SYM_FILE})
    TARGET_LINK_LIBRARIES(afopencl PUBLIC $<TARGET_FILE:afopencl_static>)
    ELSE(APPLE)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/afopencl_static.renamed
      COMMAND objcopy --redefine-syms ${BLAS_SYM_FILE} $<TARGET_FILE:afopencl_static> ${CMAKE_BINARY_DIR}/afopencl_static.renamed
      DEPENDS $<TARGET_FILE:afopencl_static>)
      TARGET_LINK_LIBRARIES(afopencl PUBLIC ${CMAKE_BINARY_DIR}/afopencl_static.renamed)
    ENDIF(APPLE)


ELSE(DEFINED BLAS_SYM_FILE)

    ADD_LIBRARY(afopencl SHARED
                ${opencl_headers}
                ${opencl_sources}
                ${jit_sources}
                ${kernel_headers}
                ${opencl_kernels}
                ${kernel_sources}
                ${conv_ker_headers}
                ${conv_ker_sources}
                ${backend_headers}
                ${backend_sources}
                ${c_headers}
                ${c_sources}
                ${cpp_sources}
                ${magma_sources}
                ${magma_headers})

ENDIF()

ADD_DEPENDENCIES(afopencl ${cl_kernel_targets})

TARGET_LINK_LIBRARIES(afopencl
                                PRIVATE ${OpenCL_LIBRARIES}
                                PRIVATE ${FreeImage_LIBS}
                                PRIVATE ${CLBLAS_LIBRARIES}
                                PRIVATE ${CLFFT_LIBRARIES}
                                PRIVATE ${CMAKE_DL_LIBS}
                                PRIVATE ${Boost_LIBRARIES})

IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
    ADD_DEPENDENCIES(afopencl forge)
ENDIF()

IF(LAPACK_FOUND)
  TARGET_LINK_LIBRARIES(afopencl
    PRIVATE ${LAPACK_LIBRARIES}
    PRIVATE ${CBLAS_LIBRARIES})
ENDIF()

SET_TARGET_PROPERTIES(afopencl PROPERTIES
    VERSION "${AF_VERSION}"
    SOVERSION "${AF_VERSION_MAJOR}")

IF(FORGE_FOUND)
    TARGET_LINK_LIBRARIES(afopencl  PRIVATE ${FORGE_LIBRARIES})
ENDIF()

INSTALL(TARGETS afopencl EXPORT OpenCL DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

IF(APPLE)
    INSTALL(SCRIPT "${CMAKE_MODULE_PATH}/osx_install/InstallTool.cmake")
ENDIF(APPLE)

export(TARGETS afopencl FILE ArrayFireOpenCL.cmake)
INSTALL(EXPORT OpenCL DESTINATION "${AF_INSTALL_CMAKE_DIR}"
    COMPONENT cmake
    FILE ArrayFireOpenCL.cmake)
