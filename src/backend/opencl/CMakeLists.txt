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
    Array.cpp
    Array.hpp
    backend.h
    copy.cpp
    copy.hpp
    ctx.cpp
    ctx.hpp
    elwise.cpp
    elwise.hpp
    print.cpp
    print.hpp
    diff.hpp
    diff.cpp
    ../generator.hpp
    ../array.cpp
    ../ArrayInfo.cpp
    ../ArrayInfo.hpp
    ../dim4.cpp
    ../elwise.cpp
    ../helper.h
    ../index.cpp
    ../print.cpp
    ../diff.cpp
    ../moddims.cpp
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
