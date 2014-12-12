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
#
FIND_PACKAGE(GLEW)
FIND_PACKAGE(OpenGL)

IF (WIN32)
    IF (NV_SYSTEM_PROCESSOR STREQUAL "AMD64")
        FIND_LIBRARY( GLEWmx_LIBRARY
            NAMES glew64mx glew64mxs
            PATHS
            $ENV{PROGRAMFILES}/GLEW/lib
            ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
            ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
            DOC "The GLEWmx library (64-bit)"
        )
    ELSE(NV_SYSTEM_PROCESSOR STREQUAL "AMD64")
        FIND_LIBRARY( GLEWmx_LIBRARY
            NAMES glewmx GLEWmx glew32mx glew32mxs
            PATHS
            $ENV{PROGRAMFILES}/GLEW/lib
            ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
            ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
            DOC "The GLEWmx library"
        )
    ENDIF(NV_SYSTEM_PROCESSOR STREQUAL "AMD64")
ELSE (WIN32)
    FIND_LIBRARY( GLEWmx_LIBRARY
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
ENDIF (WIN32)

IF (GLEW_INCLUDE_DIR AND GLEWmx_LIBRARY)
    SET(GLEWmx_FOUND "YES")
ENDIF (GLEW_INCLUDE_DIR AND GLEWmx_LIBRARY)
