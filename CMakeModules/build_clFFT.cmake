# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

INCLUDE(ExternalProject)

SET(prefix "${PROJECT_BINARY_DIR}/third_party/clFFT")
SET(clFFT_location ${prefix}/lib/import/${CMAKE_STATIC_LIBRARY_PREFIX}clFFT${CMAKE_STATIC_LIBRARY_SUFFIX})
IF(CMAKE_VERSION VERSION_LESS 3.2)
    IF(CMAKE_GENERATOR MATCHES "Ninja")
        MESSAGE(WARNING "Building clFFT with Ninja has known issues with CMake older than 3.2")
    endif()
    SET(byproducts)
ELSE()
    SET(byproducts BUILD_BYPRODUCTS ${clFFT_location})
ENDIF()

if(WIN32 AND CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR MATCHES "Ninja")
  set(extproj_gen_opts "-G${CMAKE_GENERATOR}" "-A${CMAKE_GENERATOR_PLATFORM}")
else()
  set(extproj_gen_opts "-G${CMAKE_GENERATOR}")
endif()

if("${CMAKE_BUILD_TYPE}" MATCHES "Release|RelWithDebInfo")
  set(extproj_build_type "Release")
else()
  set(extproj_build_type ${CMAKE_BUILD_TYPE})
endif()

ExternalProject_Add(
    clFFT-ext
    GIT_REPOSITORY https://github.com/arrayfire/clFFT.git
    GIT_TAG arrayfire-release
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ${extproj_gen_opts}
      -Wno-dev <SOURCE_DIR>/src
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
      -DCMAKE_BUILD_TYPE:STRING=${extproj_build_type}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_CLIENT:BOOL=OFF
      -DBUILD_TEST:BOOL=OFF
      -DSUFFIX_LIB:STRING=
    ${byproducts}
    )

ExternalProject_Get_Property(clFFT-ext install_dir)

set(CLFFT_INCLUDE_DIRS ${install_dir}/include)
make_directory(${install_dir}/include)

add_library(clFFT::clFFT IMPORTED STATIC)
set_target_properties(clFFT::clFFT PROPERTIES
  IMPORTED_LOCATION ${clFFT_location}
  INTERFACE_INCLUDE_DIRECTORIES ${install_dir}/include
  )
add_dependencies(clFFT::clFFT clFFT-ext)

set(CLFFT_LIBRARIES clFFT)
set(CLFFT_FOUND ON)
