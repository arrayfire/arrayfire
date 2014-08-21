FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
ADD_DEFINITIONS(-DAF_CUDA)
INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/backend/cuda
    )

FILE(GLOB cuda_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "*.cu"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    "kernel/*.cu"
    "kernel/*.cpp"
    "kernel/*.hpp"
    "kernel/*.h"
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xlinker -stdlib=libstdc++ -Xcompiler -fPIC")

CUDA_ADD_LIBRARY(afcuda SHARED ${cuda_src} OPTIONS "-arch=sm_20")
TARGET_LINK_LIBRARIES(afcuda ${CUDA_LIBRARIES} )
