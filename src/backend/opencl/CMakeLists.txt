INCLUDE(${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake)
ADD_DEFINITIONS(-DAF_OPENCL)

SET( cl_kernel_headers
    "kernel_headers")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/opencl
    ${OPENCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

FILE(GLOB opencl_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    )

FILE(GLOB kernel_src "kernel/*.cl")

CL_KERNEL_TO_H(
    SOURCES ${kernel_src}
    VARNAME kernel_files
    EXTENSION "hpp"
    OUTPUT_DIR ${cl_kernel_headers}
    TARGETS cl_kernel_targets
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

ADD_LIBRARY(afopencl SHARED ${opencl_src})
ADD_DEPENDENCIES(afopencl ${cl_kernel_targets})
TARGET_LINK_LIBRARIES(afopencl ${OPENCL_LIBRARIES} ${lib_deps})
