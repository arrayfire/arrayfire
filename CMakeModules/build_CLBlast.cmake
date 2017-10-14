# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

set(prefix ${PROJECT_BINARY_DIR}/third_party/CLBlast)
set(CLBlast_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clblast${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(
    CLBlast-ext
    GIT_REPOSITORY https://github.com/cnugteren/CLBlast.git
    GIT_TAG 48133a0cd1a7b61b87906ec1f4608e766e20a973
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS ${CLBlast_location}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} "-G${CMAKE_GENERATOR}" -Wno-dev <SOURCE_DIR>/
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
      -DOVERRIDE_MSVC_FLAGS_TO_MT:BOOL=OFF
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DSAMPLES:BOOL=OFF
      -DTUNERS:BOOL=OFF
      -DCLIENTS:BOOL=OFF
      -DTESTS:BOOL=OFF
      -DNETLIB:BOOL=OFF
    )

ExternalProject_Get_Property(CLBlast-ext install_dir)
set(CLBLAST_INCLUDE_DIRS ${install_dir}/include)
set(CLBLAST_LIBRARIES CLBlast)
set(CLBLAST_FOUND ON)

make_directory("${CLBLAST_INCLUDE_DIRS}")

add_library(CLBlast UNKNOWN IMPORTED)
set_target_properties(CLBlast PROPERTIES
  IMPORTED_LOCATION "${CLBlast_location}"
  INTERFACE_INCLUDE_DIRECTORIES "${CLBLAST_INCLUDE_DIRS}")
add_dependencies(CLBlast CLBlast-ext)
