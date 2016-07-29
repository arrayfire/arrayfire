INCLUDE(ExternalProject)

SET(prefix ${CMAKE_BINARY_DIR}/third_party/cl2hpp)

ExternalProject_Add(
    cl2hpp-ext
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
    GIT_TAG 75bb7d0d8b2ffc6aac0a3dcaa22f6622cab81f7c
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}/package"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>
    -DCMAKE_SOURCE_DIR:PATH=<SOURCE_DIR>
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    )

ExternalProject_Get_Property(cl2hpp-ext install_dir)

ADD_CUSTOM_TARGET(cl2hpp DEPENDS "${prefix}/package/CL/cl2.hpp")

ADD_DEPENDENCIES(cl2hpp cl2hpp-ext)

SET(CL2HPP_INCLUDE_DIRECTORY ${install_dir})
