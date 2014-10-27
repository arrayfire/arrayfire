INCLUDE(${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake)
ADD_DEFINITIONS(-DAF_OPENCL)
ADD_DEFINITIONS( -D__CL_ENABLE_EXCEPTIONS
                 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS )

#INCLUDE(${CMAKE_SOURCE_DIR}/common/FindCLBLAS.cmake)
FIND_PACKAGE(CLBLAS REQUIRED)

FIND_PACKAGE(clFFT REQUIRED)
FIND_PACKAGE(Boost 1.48 REQUIRED)
FIND_PACKAGE(BoostCompute REQUIRED)

SET( cl_kernel_headers
    "kernel_headers")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/opencl
    ${OPENCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CLBLAS_INCLUDE_DIR}
    ${CLFFT_INCLUDE_DIRS}
    ${BoostCompute_INCLUDE_DIR}
    )

FILE(GLOB opencl_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "kernel/*.hpp"
    "kernel/*.cl"
    "jit/*.hpp"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    "kernel/*.cpp"
    )

FILE(GLOB kernel_src "kernel/*.cl" "kernel/KParam.hpp")

CL_KERNEL_TO_H(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${cl_kernel_headers}
    TARGETS cl_kernel_targets
    NAMESPACE "opencl"
    EOF "0"
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread")

LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})
ADD_LIBRARY(afopencl SHARED ${opencl_src})
ADD_DEPENDENCIES(afopencl ${cl_kernel_targets})
TARGET_LINK_LIBRARIES(afopencl ${OPENCL_LIBRARIES} ${lib_deps} ${CLBLAS_LIBRARIES} ${CLFFT_LIBRARIES} ${CMAKE_DL_LIBS})
