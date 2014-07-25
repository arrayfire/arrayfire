INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    )

SET(cpp_src
    elwise.cpp elwise.hpp
    copy.cpp
    index.cpp
    Array.cpp Array.hpp
    generator.cpp
    print.cpp
    backend_helper.h
    ../ArrayInfo.cpp
    ../ArrayInfo.hpp
    ../common_helper.h
    ../elwise.cpp
    ../generator.cpp
    ../copy.cpp
    ../index.cpp
    ../print.cpp
    ../dim4.cpp
    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ADD_LIBRARY(afcpu SHARED ${cpp_src})
TARGET_LINK_LIBRARIES(afcpu)
