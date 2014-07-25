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

SET( opencl_src
    Array.cpp Array.hpp
    ctx.cpp ctx.hpp
    elwise.cpp elwise.hpp
    copy.cpp
    index.cpp
    print.cpp
    generator.cpp
    backend_helper.h
    ../ArrayInfo.cpp
    ../ArrayInfo.hpp
    ../common_helper.h
    ../elwise.cpp
    ../generator.cpp
    ../copy.cpp
    ../index.cpp
    ../dim4.cpp
    ../print.cpp
    )

SET ( kernel_src
    kernel/set.cl
    kernel/binaryOp.cl
    )

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
TARGET_LINK_LIBRARIES(afopencl ${OPENCL_LIBRARIES})
