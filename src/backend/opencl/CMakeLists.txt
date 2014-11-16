CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE("${CMAKE_SOURCE_DIR}/common/CLKernelToH.cmake")

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_OPENCL)
ADD_DEFINITIONS( -D__CL_ENABLE_EXCEPTIONS
                 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS )

# clBLAS and clFFT create packages when their `make install` step is executed,
# rather than installing to system directories. Thus we have special find
# and install methods for these libraries.
FIND_PACKAGE(CLBLAS REQUIRED)
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})
# Custom install command
INSTALL(DIRECTORY "${CLBLAS_PACKAGE_DIR}/include/" DESTINATION include)
FILE(GLOB CLBLAS_LIBS "${CLBLAS_PACKAGE_DIR}/lib64/*.so*")
INSTALL(FILES ${CLBLAS_LIBS} DESTINATION lib)

FIND_PACKAGE(clFFT REQUIRED)
# Custom install command
INSTALL(DIRECTORY "${CLFFT_BUILD_DIR}/package/include/" DESTINATION include)
FILE(GLOB CLFFT_LIBS "${CLFFT_BUILD_DIR}/package/lib64/*.so*")
INSTALL(FILES ${CLFFT_LIBS} DESTINATION lib)

ADD_DEFINITIONS( -DBOOST_ALL_NO_LIB )
SET(Boost_USE_STATIC_LIBS OFF)
FIND_PACKAGE(Boost 1.48 COMPONENTS chrono REQUIRED)

FIND_PACKAGE(BoostCompute REQUIRED)

SET( cl_kernel_headers
    "kernel_headers")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend/opencl"
    ${OPENCL_INCLUDE_DIRS}
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CLBLAS_INCLUDE_DIR}
    ${CLFFT_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
    ${BoostCompute_INCLUDE_DIR}
    )

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )
FILE(GLOB backend_sources
    "../*.cpp"
    )
FILE(GLOB cpp_sources
    "../../frontend/cpp/*.cpp"
    )
source_group(Headers FILES ${backend_headers})
source_group(Sources FILES ${backend_sources})
source_group(CPP\ Sources FILES ${cpp_sources})

FILE(GLOB opencl_headers
	"*.hpp"
    "*.h")
FILE(GLOB opencl_sources
    "*.cpp")
source_group(opencl\\Headers FILES ${opencl_headers})
source_group(opencl\\Sources FILES ${opencl_sources})

FILE(GLOB jit_sources
    "jit/*.hpp")
source_group(opencl\\JIT FILES ${jit_sources})

FILE(GLOB kernel_headers
    "kernel/*.hpp")
FILE(GLOB opencl_kernels
    "kernel/*.cl")
FILE(GLOB kernel_sources
     "kernel/*.cpp")
source_group(opencl\\kernel\\Headers FILES ${kernel_headers})
source_group(opencl\\kernel\\cl FILES ${opencl_kernels})
source_group(opencl\\kernel\\Sources FILES ${kernel_sources})

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
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread")
ELSE(${UNIX}) #Windows
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()


ADD_LIBRARY(afopencl SHARED ${cpp_sources}
							${backend_headers}
							${backend_sources}
							${opencl_headers}
							${opencl_sources}
							${jit_sources}
							${kernel_headers}
							${opencl_kernels}
							${kernel_sources})
							
ADD_DEPENDENCIES(afopencl ${cl_kernel_targets})

TARGET_LINK_LIBRARIES(afopencl ${OPENCL_LIBRARIES} ${lib_deps} ${CLBLAS_LIBRARIES} ${CLFFT_LIBRARIES} ${CMAKE_DL_LIBS} ${Boost_LIBRARIES})

INSTALL(TARGETS afopencl DESTINATION lib)
