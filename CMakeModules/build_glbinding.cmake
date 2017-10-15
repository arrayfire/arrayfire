# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

INCLUDE(ExternalProject)

SET(prefix ${PROJECT_BINARY_DIR}/third_party/glb)

SET(LIB_POSTFIX "")
IF (CMAKE_BUILD_TYPE MATCHES Debug)
    SET(LIB_POSTFIX "d")
ENDIF()

SET(glbinding_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glbinding${LIB_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})

IF(CMAKE_VERSION VERSION_LESS 3.2)
    IF(CMAKE_GENERATOR MATCHES "Ninja")
        MESSAGE(WARNING "Building with Ninja has known issues with CMake older than 3.2")
    endif()
    SET(byproducts)
ELSE()
    SET(byproducts BUILD_BYPRODUCTS ${glbinding_location})
ENDIF()

IF(UNIX)
    SET(CXXFLAGS "${CMAKE_CXX_FLAGS} -w -fPIC")
    SET(CFLAGS "${CMAKE_C_FLAGS} -w -fPIC")
ENDIF(UNIX)

ExternalProject_Add(
    glb-ext
    GIT_REPOSITORY https://github.com/cginternals/glbinding.git
    GIT_TAG v2.1.1
    UPDATE_COMMAND ""
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CXXFLAGS}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${CFLAGS}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DOPTION_BUILD_TESTS:BOOL=OFF
    # Leave these GLFW_LIBRARY Options as empty.
    # They are only required for GLBINDING Executables.
    # Leaving them empty will disable compilation of said executables.
    -DGLFW_LIBRARY_RELEASE:PATH=
    -DGLFW_LIBRARY_DEBUG:PATH=
    ${byproducts}
    )

ADD_LIBRARY(glbinding IMPORTED STATIC)

ExternalProject_Get_Property(glb-ext install_dir)

SET_TARGET_PROPERTIES(glbinding PROPERTIES IMPORTED_LOCATION ${glbinding_location})

ADD_DEPENDENCIES(glbinding glb-ext)

SET(GLBINDING_INCLUDE_DIRS ${install_dir}/include CACHE INTERNAL "" FORCE)
SET(GLBINDING_LIBRARIES ${glbinding_location} CACHE INTERNAL "" FORCE)
# Use glbinding_DIR as is and don't change the case
SET(glbinding_DIR ${install_dir} CACHE INTERNAL "" FORCE)
SET(GLBINDING_FOUND ON CACHE INTERNAL "" FORCE)
