INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    )

SET(cpp_src
    Array.cpp
    Array.hpp
    backend.h
    copy.cpp
    copy.hpp
    elwise.cpp
    elwise.hpp
    generator.cpp
    generator.hpp
    index.cpp
    index.hpp
    print.cpp
    print.hpp
    diff.hpp
    diff.cpp
    moddims.cpp
    moddims.hpp
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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ADD_LIBRARY(afcpu SHARED ${cpp_src})
TARGET_LINK_LIBRARIES(afcpu)
