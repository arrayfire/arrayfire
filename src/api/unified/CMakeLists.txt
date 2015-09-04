
FILE(GLOB hapi_headers
    "*.hpp"
    "*.h")

FILE(GLOB hapi_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

FILE(GLOB backend_sources
    "../c/util.cpp")

ADD_LIBRARY(af SHARED
            ${hapi_headers}
            ${hapi_sources})

IF(${BUILD_CPU})
    ADD_DEPENDENCIES(af afcpu)
ENDIF()

IF(${BUILD_CUDA})
    ADD_DEPENDENCIES(af afcuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_DEPENDENCIES(af afopencl)
ENDIF()

SET_TARGET_PROPERTIES(af PROPERTIES
                      VERSION "${AF_VERSION}"
                      SOVERSION "${AF_VERSION_MAJOR}")

INSTALL(TARGETS af EXPORT AF DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

IF(APPLE)
    INSTALL(SCRIPT "${CMAKE_MODULE_PATH}/osx_install/InstallTool.cmake")
ENDIF(APPLE)

EXPORT(TARGETS af FILE ArrayFireHAPI.cmake)
INSTALL(EXPORT AF DESTINATION "${AF_INSTALL_CMAKE_DIR}"
        COMPONENT cmake
        FILE ArrayFireHAPI.cmake)
