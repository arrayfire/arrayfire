
FIND_PACKAGE(BLAS REQUIRED)
MESSAGE(STATUS "BLAS Library: ${BLAS_LIBRARIES}")

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    ${BLAS_INCLUDE_DIR}
    )

FILE(GLOB cpp_src
    "*.hpp"
    "*.h"
    "*.cpp"
    "../*.hpp"
    "../*.h"
    "../*.cpp"
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ADD_LIBRARY(afcpu SHARED ${cpp_src})
TARGET_LINK_LIBRARIES(afcpu ${lib_deps} ${BLAS_LIBRARIES})
