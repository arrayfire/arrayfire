# Source from
#https://github.com/LaurentGomila/SFML/blob/master/cmake/Modules/FindGLEW.cmake

#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
# GLEWmx_LIBRARY
# GLEWmxd_LIBRARY
# GLEWmxs_LIBRARY

FIND_PACKAGE(OpenGL REQUIRED)

OPTION(USE_GLEWmx_STATIC "Use Static GLEWmx Lib" OFF)

FIND_PATH(GLEW_INCLUDE_DIR GL/glew.h
    HINTS
    ${GLEW_ROOT_DIR}/include
    )

IF (WIN32)
    FIND_LIBRARY( GLEWmxd_LIBRARY
        NAMES glewmx GLEWmx glew32mx glew32mx
        PATHS
        $ENV{PROGRAMFILES}/GLEW/lib
        ${GLEW_ROOT_DIR}/lib
        ${GLEW_ROOT_DIR}
        ${PROJECT_SOURCE_DIR}/../dependencies/glew/lib
        PATH_SUFFIXES "Release MX/x64" "lib64"
        DOC "The GLEWmx library"
    )
    FIND_LIBRARY( GLEWmxs_LIBRARY
        NAMES glewmxs GLEWmxs glew32mxs glew32mxs
        PATHS
        $ENV{PROGRAMFILES}/GLEW/lib
        ${GLEW_ROOT_DIR}/lib
        ${GLEW_ROOT_DIR}
        ${PROJECT_SOURCE_DIR}/../dependencies/glew/lib
        PATH_SUFFIXES "Release MX/x64" "lib64"
        DOC "The GLEWmxs Static library"
    )
ELSE (WIN32)
    FIND_LIBRARY( GLEWmxd_LIBRARY
        NAMES GLEWmx glewmx
        PATHS
        /usr/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/arm-linux-gnueabihf
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        ${GLEW_ROOT_DIR}/lib
        NO_DEFAULT_PATH
        DOC "The GLEWmx library")

    SET(PX ${CMAKE_STATIC_LIBRARY_PREFIX})
    SET(SX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    FIND_LIBRARY( GLEWmxs_LIBRARY
        NAMES ${PX}GLEWmx${SX} ${PX}glewmx${SX}
        PATHS
        /usr/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/arm-linux-gnueabihf
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        ${GLEW_ROOT_DIR}/lib
        NO_DEFAULT_PATH
        DOC "The GLEWmx library")
    UNSET(PX)
    UNSET(SX)
ENDIF (WIN32)

IF(USE_GLEWmx_STATIC)
    MESSAGE(STATUS "Using Static GLEWmx Lib")
    ADD_DEFINITIONS(-DGLEW_STATIC)
    SET(GLEWmx_LIBRARY ${GLEWmxs_LIBRARY})
ELSE(USE_GLEWmx_STATIC)
    MESSAGE(STATUS "Using Dynamic GLEWmx Lib")
    REMOVE_DEFINITIONS(-DGLEW_STATIC)
    SET(GLEWmx_LIBRARY ${GLEWmxd_LIBRARY})
ENDIF(USE_GLEWmx_STATIC)

IF (GLEW_INCLUDE_DIR AND GLEWmx_LIBRARY)
    SET(GLEWmx_FOUND "YES")
ENDIF (GLEW_INCLUDE_DIR AND GLEWmx_LIBRARY)
