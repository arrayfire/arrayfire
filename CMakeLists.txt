CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(ARRAYFIRE)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
if(POLICY CMP0058)
  CMAKE_POLICY(SET CMP0058 NEW)
endif(POLICY CMP0058)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
INCLUDE(UploadCoveralls)
INCLUDE(AFInstallDirs)
INCLUDE(platform)

OPTION(BUILD_TEST "Build Tests" ON)
OPTION(BUILD_EXAMPLES "Build Examples" ON)

OPTION(BUILD_CPU "Build ArrayFire with a CPU backend" ON)

FIND_PACKAGE(CUDA QUIET)
IF(${CUDA_FOUND})
    SET(BUILD_CUDA ON CACHE BOOL "")
ENDIF(${CUDA_FOUND})
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)

FIND_PACKAGE(OpenCL QUIET)
IF(${OpenCL_FOUND})
    SET(BUILD_OPENCL ON CACHE BOOL "")
ENDIF(${OpenCL_FOUND})
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

OPTION(BUILD_GRAPHICS "Build ArrayFire with Forge Graphics" ON)

OPTION(BUILD_DOCS "Create ArrayFire Documentation" OFF)
OPTION(WITH_COVERAGE "Added code coverage flags" OFF)

OPTION(BUILD_NONFREE "Build ArrayFire nonfree algorithms" OFF)

OPTION(BUILD_UNIFIED "Build Backend-Independent ArrayFire API" ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
      "MinSizeRel" "RelWithDebInfo")
endif()

OPTION(MIN_BUILD_TIME "This flag compiles ArrayFire with O0, which is the fastest way to compile" OFF)
INCLUDE(MinBuildTime)

FIND_PACKAGE(FreeImage)
IF(FREEIMAGE_FOUND)
    ADD_DEFINITIONS(-DWITH_FREEIMAGE)
    SET(FreeImage_LIBS ${FREEIMAGE_LIBRARY})
    INCLUDE_DIRECTORIES(BEFORE ${FREEIMAGE_INCLUDE_PATH})
ELSE(FREEIMAGE_FOUND)
    MESSAGE(WARNING, "FreeImage not found!")
ENDIF(FREEIMAGE_FOUND)

OPTION(USE_SYSTEM_CL2HPP "Use cl2.hpp installed on system" OFF)

IF(BUILD_GRAPHICS)
    OPTION(USE_SYSTEM_FORGE "Use system Forge" OFF)
    OPTION(USE_SYSTEM_GLBINDING "Use system glbinding" OFF)

    FIND_PACKAGE(OpenGL REQUIRED)

    IF(USE_SYSTEM_GLBINDING)
        # Point glbinding_DIR (case sensitive) to the location of glbinding-config.cmake
        # This file is generally at CMAKE_INSTALL_PREFIX/glbinding-config.cmake of
        # the glbinding project
        FIND_PACKAGE(glbinding REQUIRED)
        SET(GLBINDING_FOUND "ON")
        SET(GLBINDING_LIBRARIES glbinding::glbinding CACHE INTERNAL "glbinding library target")
    ELSE(USE_SYSTEM_GLBINDING)
        INCLUDE(build_glbinding)
        LIST(APPEND GRAPHICS_DEPENDENCIES glbinding)
    ENDIF(USE_SYSTEM_GLBINDING)

    IF(USE_SYSTEM_FORGE)
      FIND_PACKAGE(Forge REQUIRED)
    ELSE(USE_SYSTEM_FORGE)  # Build Forge as an external Project
        INCLUDE(build_forge)
        LIST(APPEND GRAPHICS_DEPENDENCIES forge)
    ENDIF(USE_SYSTEM_FORGE)

    IF(FORGE_FOUND AND GLBINDING_FOUND)
        ADD_DEFINITIONS(-DWITH_GRAPHICS)

        INCLUDE_DIRECTORIES(BEFORE
            ${FORGE_INCLUDE_DIRS}
            ${GLBINDING_INCLUDE_DIRS})

        SET(GRAPHICS_LIBRARIES  ${FORGE_LIBRARIES}
                                ${GLBINDING_LIBRARIES}
                                ${OPENGL_gl_LIBRARY})

        IF(APPLE)
            FIND_PACKAGE(X11 REQUIRED)
            INCLUDE_DIRECTORIES(BEFORE ${X11_INCLUDE_DIR})
        ENDIF(APPLE)

    ELSE(FORGE_FOUND AND GLBINDING_FOUND)
        MESSAGE(WARNING "Graphics dependencies (Forge and/or glbinding) not found. Graphics will be disabled")
    ENDIF(FORGE_FOUND AND GLBINDING_FOUND)

ENDIF(BUILD_GRAPHICS)

IF(${BUILD_NONFREE})
    MESSAGE(WARNING "Building With NONFREE ON requires the following patents")
    SET(BUILD_NONFREE_SIFT ON CACHE BOOL "Build ArrayFire with SIFT")
    MARK_AS_ADVANCED(BUILD_NONFREE_SIFT)
ELSE(${BUILD_NONFREE})
    UNSET(BUILD_NONFREE_SIFT CACHE) # BUILD_NONFREE_SIFT cannot be built without BUILD_NONFREE
ENDIF(${BUILD_NONFREE})

IF(${BUILD_NONFREE_SIFT})
  ADD_DEFINITIONS(-DAF_BUILD_NONFREE_SIFT)

  MESSAGE(WARNING "Building with SIFT requires the following patents")

  MESSAGE("Method and apparatus for identifying scale invariant features"
    "in an image and use of same for locating an object in an image,\" David"
    "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application"
    "filed March 8, 1999. Asignee: The University of British Columbia. For"
    "further details, contact David Lowe (lowe@cs.ubc.ca) or the"
    "University-Industry Liaison Office of the University of British"
    "Columbia.")
ENDIF(${BUILD_NONFREE_SIFT})

INCLUDE_DIRECTORIES(BEFORE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/c"
    )

IF(${UNIX})
    ADD_DEFINITIONS(-std=c++11 -Wall -fvisibility=hidden)

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
    IF(CMAKE_GENERATOR MATCHES "Ninja")
        ADD_COMPILE_OPTIONS(/FS)
    ENDIF()
    IF(MSVC)
        # MP is multiprocess compilation. Gm- disables minimal rebuilds
        # http://stackoverflow.com/questions/6172205/how-can-i-do-a-parallel-build-in-visual-studio-2010vvvvvvvv
        # http://www.kitware.com/blog/home/post/434
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Gm- /bigobj")
        SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /MP /Gm-")
    ENDIF(MSVC)
ENDIF()

# Architechture Definitions
INCLUDE(TargetArch)
target_architecture(ARCH)

INCLUDE(Version)

IF(${BUILD_CPU})
    ADD_SUBDIRECTORY(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/backend/opencl)
ENDIF()

IF(${BUILD_UNIFIED})
    ADD_DEFINITIONS(-DAF_UNIFIED)
    ADD_SUBDIRECTORY(src/api/unified)
ENDIF()

IF(${BUILD_DOCS})
    ADD_SUBDIRECTORY(docs)
ENDIF()

ADD_EXECUTABLE(bin2cpp ${PROJECT_SOURCE_DIR}/CMakeModules/bin2cpp.cpp)

IF(${BUILD_TEST})
    INCLUDE(CTest)
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
    ${PROJECT_SOURCE_DIR}/include/af/version.h DESTINATION "${AF_INSTALL_INC_DIR}/af/"
    COMPONENT headers
)

IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
    OPTION(INSTALL_FORGE_DEV "Install Forge Header and Share Files with ArrayFire" OFF)
    INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/third_party/forge/lib/"
            DESTINATION "${AF_INSTALL_LIB_DIR}"
            COMPONENT libraries
    )
    IF(${INSTALL_FORGE_DEV})
        INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/third_party/forge/include/"
                DESTINATION "${AF_INSTALL_INC_DIR}"
                COMPONENT headers
        )
        INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/third_party/forge/share/Forge/"
                DESTINATION "${AF_INSTALL_DATA_DIR}/../Forge"
                COMPONENT share
        )
    ENDIF(${INSTALL_FORGE_DEV})
ENDIF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)

## configuration to be used from the binary directory directly
SET(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(BACKEND_DIR "src/backend/\${lowerbackend}")
SET(UNIFIED_DIR "src/api/unified")
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/CMakeModules/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfig.cmake
    @ONLY)

## installed cmake configuration
# use a relative dir to keep arrayfire relocatable
STRING(REGEX REPLACE "[^/]+" ".." reldir "${AF_INSTALL_CMAKE_DIR}")
SET(INCLUDE_DIR "\${CMAKE_CURRENT_LIST_DIR}/${reldir}/include")
set(BACKEND_DIR)
set(UNIFIED_DIR)
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/CMakeModules/ArrayFireConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    @ONLY)
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/CMakeModules/ArrayFireConfigVersion.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/Install/ArrayFireConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ArrayFireConfigVersion.cmake
    DESTINATION ${AF_INSTALL_CMAKE_DIR}
    COMPONENT cmake)

# install the examples irrespective of the BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on BUILD_EXAMPLES
# when BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
    DESTINATION "${AF_INSTALL_EXAMPLE_DIR}"
    COMPONENT examples)

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets/examples"
    DESTINATION "${AF_INSTALL_EXAMPLE_DIR}/assets"
    COMPONENT examples)

IF(APPLE)
    INCLUDE(osx_install/OSXInstaller)
ENDIF(APPLE)

##
# Packaging
##
include(CPackConfig)
