# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

FetchContent_Declare(
  ${clblast_prefix}
  GIT_REPOSITORY    https://github.com/cnugteren/CLBlast.git
  GIT_TAG           1.5.2
)
af_dep_check_and_populate(${clblast_prefix})

include(ExternalProject)
find_program(GIT git)

set(prefix ${PROJECT_BINARY_DIR}/third_party/CLBlast)
set(CLBlast_libname ${CMAKE_STATIC_LIBRARY_PREFIX}clblast${CMAKE_STATIC_LIBRARY_SUFFIX})
set(CLBlast_location ${${clblast_prefix}_BINARY_DIR}/pkg/lib/${CLBlast_libname})

set(extproj_gen_opts "-G${CMAKE_GENERATOR}")
if(WIN32 AND CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR MATCHES "Ninja")
  list(APPEND extproj_gen_opts "-A${CMAKE_GENERATOR_PLATFORM}")
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND extproj_gen_opts "-T${CMAKE_GENERATOR_TOOLSET}")
  endif()
endif()
if(VCPKG_TARGET_TRIPLET)
  list(APPEND extproj_gen_opts "-DOPENCL_ROOT:PATH=${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
endif()

set(extproj_build_type_option "")
if(NOT isMultiConfig)
  if("${CMAKE_BUILD_TYPE}" MATCHES "Release|RelWithDebInfo")
    set(extproj_build_type "Release")
  else()
    set(extproj_build_type ${CMAKE_BUILD_TYPE})
  endif()
  set(extproj_build_type_option "-DCMAKE_BUILD_TYPE:STRING=${extproj_build_type}")
endif()

ExternalProject_Add(
    CLBlast-ext
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${${clblast_prefix}_SOURCE_DIR}"
    BINARY_DIR "${${clblast_prefix}_BINARY_DIR}"
    PREFIX "${prefix}"
    INSTALL_DIR "${${clblast_prefix}_BINARY_DIR}/pkg"
    BUILD_BYPRODUCTS ${CLBlast_location}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ${extproj_gen_opts}
      -Wno-dev <SOURCE_DIR>
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
      -DOVERRIDE_MSVC_FLAGS_TO_MT:BOOL=OFF
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
      ${extproj_build_type_option}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_LIBDIR:PATH=lib
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DSAMPLES:BOOL=OFF
      -DTUNERS:BOOL=OFF
      -DCLIENTS:BOOL=OFF
      -DTESTS:BOOL=OFF
      -DNETLIB:BOOL=OFF
    )

set(CLBLAST_INCLUDE_DIRS "${${clblast_prefix}_BINARY_DIR}/pkg/include")
set(CLBLAST_LIBRARIES CLBlast)
set(CLBLAST_FOUND ON)

make_directory("${CLBLAST_INCLUDE_DIRS}")

add_library(CLBlast UNKNOWN IMPORTED)
set_target_properties(CLBlast PROPERTIES
  IMPORTED_LOCATION "${CLBlast_location}"
  INTERFACE_INCLUDE_DIRECTORIES "${CLBLAST_INCLUDE_DIRS}")

add_dependencies(CLBlast CLBlast-ext)
