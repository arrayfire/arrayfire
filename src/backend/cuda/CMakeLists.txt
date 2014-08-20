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
    Array.cu
    Array.hpp
    backend.h
    copy.cu
    copy.hpp
    elwise.cu
    elwise.hpp
    print.cu
    print.hpp
    diff.hpp
    diff.cu
    random.hpp
    random.cu
    ../generator.hpp
    transpose.cu
    transpose.hpp
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
    ../transpose.cpp
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xlinker -stdlib=libstdc++ -Xcompiler -fPIC")

CUDA_ADD_LIBRARY(afcuda SHARED ${cuda_src} OPTIONS "-arch=sm_20")
TARGET_LINK_LIBRARIES(afcuda ${CUDA_LIBRARIES} )
