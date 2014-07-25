FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
ADD_DEFINITIONS(-DAF_CUDA)
INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/backend/cuda
    )

SET( cuda_src
    kernel/elwise.cu
    Array.cu Array.hpp
    elwise.cu elwise.hpp
    copy.cu
    print.cu
    index.cu index.hpp
    generator.cu
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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xlinker -stdlib=libstdc++ -Xcompiler -fPIC")

CUDA_ADD_LIBRARY(afcuda SHARED ${cuda_src} OPTIONS "-arch=sm_20")
TARGET_LINK_LIBRARIES(afcuda ${CUDA_LIBRARIES} )
