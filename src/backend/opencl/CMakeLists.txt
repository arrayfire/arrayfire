INCLUDE(${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake)
ADD_DEFINITIONS(-DAF_OPENCL)

#INCLUDE(${CMAKE_SOURCE_DIR}/common/FindCLBLAS.cmake)
FIND_PACKAGE(CLBLAS REQUIRED)

FIND_PACKAGE(clFFT REQUIRED)

SET( cl_kernel_headers
    "kernel_headers")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/opencl
    ${OPENCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CLBLAS_INCLUDE_DIR}
    ${CLFFT_INCLUDE_DIRS}
    )

FILE(GLOB opencl_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    )

FILE(GLOB kernel_src "kernel/*.cl" "kernel/KParam.hpp")

CL_KERNEL_TO_H(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${cl_kernel_headers}
    TARGETS cl_kernel_targets
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})
ADD_LIBRARY(afopencl SHARED ${opencl_src})
ADD_DEFINITIONS( -D__CL_ENABLE_EXCEPTIONS )
ADD_DEPENDENCIES(afopencl ${cl_kernel_targets})
TARGET_LINK_LIBRARIES(afopencl ${OPENCL_LIBRARIES} ${lib_deps} ${CLBLAS_LIBRARIES} ${CLFFT_LIBRARIES})
