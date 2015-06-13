CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ARRAYFIRE)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_GTEST "Download gtest and check for updates. Necessary if you change compilers" ON)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

OPTION(BUILD_GRAPHICS "Build ArrayFire with Forge Graphics" ON)

OPTION(BUILD_DOCS "Create ArrayFire Documentation" OFF)
OPTION(WITH_COVERAGE "Added code coverage flags" OFF)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
      "MinSizeRel" "RelWithDebInfo")
endif()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
INCLUDE(${CMAKE_MODULE_PATH}/UploadCoveralls.cmake)
INCLUDE(AFInstallDirs)

FIND_PACKAGE(FreeImage)
IF(FREEIMAGE_FOUND)
    ADD_DEFINITIONS(-DWITH_FREEIMAGE)
    SET(FreeImage_LIBS ${FREEIMAGE_LIBRARY})
    MESSAGE(STATUS "Using FreeImage library ${FreeImage_LIBS}")
    INCLUDE_DIRECTORIES(${FREEIMAGE_INCLUDE_PATH})
ELSE(FREEIMAGE_FOUND)
    MESSAGE(WARNING, "FreeImage not found!")
ENDIF(FREEIMAGE_FOUND)

IF(BUILD_GRAPHICS)
    OPTION(USE_SYSTEM_FORGE "Use system Forge" OFF)
    IF(USE_SYSTEM_FORGE)
        FIND_PACKAGE(Forge)
    ELSE(USE_SYSTEM_FORGE)
        INCLUDE("${CMAKE_MODULE_PATH}/build_forge.cmake")
    ENDIF(USE_SYSTEM_FORGE)

    IF(FORGE_FOUND)
        ADD_DEFINITIONS(-DGLEW_MX -DWITH_GRAPHICS)
        INCLUDE("${CMAKE_MODULE_PATH}/FindGLEWmx.cmake")

        INCLUDE_DIRECTORIES(
            ${FORGE_INCLUDE_DIRECTORIES}
            ${GLEW_INCLUDE_DIR}
            )

        SET(FORGE_LIBRARIES ${FORGE_LIBRARIES}
                            ${GLEWmx_LIBRARY}
                            ${OPENGL_gl_LIBRARY}
                            ${OPENGL_glu_LIBRARY})

        IF(APPLE)
            FIND_PACKAGE(X11 REQUIRED)
            INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR})
        ENDIF(APPLE)

    ELSE(FORGE_FOUND)
        MESSAGE(WARNING "Forge not found. Graphics will be disabled")
    ENDIF(FORGE_FOUND)

ENDIF(BUILD_GRAPHICS)

INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/c"
    )

IF(${UNIX})
    ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)
    IF(${WITH_COVERAGE})
        SET(CMAKE_CXX_FLAGS             "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_EXE_LINKER_FLAGS      "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_SHARED_LINKER_FLAGS   "-fprofile-arcs -ftest-coverage")
        SET(CMAKE_STATIC_LINKER_FLAGS   "-fprofile-arcs -ftest-coverage")
    ENDIF(${WITH_COVERAGE})
ENDIF(${UNIX})

# OS Definitions
IF(UNIX)
    IF(APPLE)   #OSX
        ADD_DEFINITIONS(-DOS_MAC)

        SET(CMAKE_MACOSX_RPATH ON)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}")
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

        LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}" isSystemDir)
        IF("${isSystemDir}" STREQUAL "-1")
            SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}")
        ENDIF("${isSystemDir}" STREQUAL "-1")
    ELSE(APPLE) #Linux
        ADD_DEFINITIONS(-DOS_LNX)
    ENDIF()
ELSE(${UNIX}) #Windows
    ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF()

# Architechture Definitions
INCLUDE(${CMAKE_MODULE_PATH}/TargetArch.cmake)
target_architecture(ARCH)
IF(${ARCH} STREQUAL "x86_64")
    ADD_DEFINITIONS(-DARCH_64)
ELSE(${ARCH})
    ADD_DEFINITIONS(-DARCH_32)
ENDIF()

INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

IF(${BUILD_CPU})
    ADD_SUBDIRECTORY(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/backend/opencl)
ENDIF()

IF(${BUILD_DOCS})
    ADD_SUBDIRECTORY(docs)
ENDIF()

ADD_EXECUTABLE(bin2cpp ${CMAKE_MODULE_PATH}/bin2cpp.cpp)

IF(${BUILD_TEST})
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF()

IF(${BUILD_EXAMPLES})
    ADD_SUBDIRECTORY(examples)
ENDIF()

##
# Installation of headers, and CMake scripts
##
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${AF_INSTALL_INC_DIR}"
    COMPONENT headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)

## The ArrayFire version file is generated and won't be included above, install
## it separately.
INSTALL(FILES
    ${CMAKE_SOURCE_DIR}/include/af/version.h DESTINATION "${AF_INSTALL_INC_DIR}/af/"
    COMPONENT headers
)

IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
    INSTALL(DIRECTORY "${CMAKE_BINARY_DIR}/third_party/forge/lib/" DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries
    )
ENDIF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)

## configuration to be used from the binary directory directly
SET(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(BACKEND_DIR "src/backend/\${lowerbackend}")
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfig.cmake
    @ONLY)

## installed cmake configuration
# use a relative dir to keep arrayfire relocatable
STRING(REGEX REPLACE "[^/]+" ".." reldir "${AF_INSTALL_CMAKE_DIR}")
SET(INCLUDE_DIR "\${CMAKE_CURRENT_LIST_DIR}/${reldir}/include")
set(BACKEND_DIR)
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    @ONLY)
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireConfigVersion.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    DESTINATION ${AF_INSTALL_CMAKE_DIR}
    COMPONENT cmake)

IF(APPLE)
    INCLUDE("${CMAKE_MODULE_PATH}/osx_install/OSXInstaller.cmake")
ENDIF(APPLE)

##
# Packaging
##
include(${CMAKE_CURRENT_SOURCE_DIR}/CPack.cmake)
