INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    )

FILE(GLOB cpp_src
    "*.h*"
    "*.c*"
    "../*.h*"
    "../*.c*"
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ADD_LIBRARY(afcpu SHARED ${cpp_src})
TARGET_LINK_LIBRARIES(afcpu)
