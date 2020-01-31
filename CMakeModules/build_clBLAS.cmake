# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

set(prefix ${PROJECT_BINARY_DIR}/third_party/clBLAS)
set(clBLAS_location ${prefix}/lib/import/${CMAKE_STATIC_LIBRARY_PREFIX}clBLAS${CMAKE_STATIC_LIBRARY_SUFFIX})

find_package(OpenCL)

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES "Ninja")
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
    clBLAS-ext
    GIT_REPOSITORY https://github.com/arrayfire/clBLAS.git
    GIT_TAG arrayfire-release
    BUILD_BYPRODUCTS ${clBLAS_location}
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ${extproj_gen_opts}
      -Wno-dev <SOURCE_DIR>/src
      -DCMAKE_CXX_FLAGS:STRING="-fPIC"
      -DCMAKE_C_FLAGS:STRING="-fPIC"
      -DCMAKE_BUILD_TYPE:STRING=${extproj_build_type}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DBUILD_CLIENT:BOOL=OFF
      -DBUILD_TEST:BOOL=OFF
      -DBUILD_KTEST:BOOL=OFF
      -DSUFFIX_LIB:STRING=

      # clBLAS uses a custom FindOpenCL that doesn't work well on Ubuntu
      -DOPENCL_LIBRARIES:FILEPATH=${OpenCL_LIBRARIES}
    )

ExternalProject_Get_Property(clBLAS-ext install_dir)

set(CLBLAS_INCLUDE_DIRS ${install_dir}/include)
set(CLBLAS_LIBRARIES clBLAS::clBLAS)
set(CLBLAS_FOUND ON)
make_directory("${CLBLAS_INCLUDE_DIRS}")

add_library(clBLAS::clBLAS UNKNOWN IMPORTED)
set_target_properties(clBLAS::clBLAS PROPERTIES
  IMPORTED_LOCATION "${clBLAS_location}"
  INTERFACE_INCLUDE_DIRECTORIES "${CLBLAS_INCLUDE_DIRS}")
add_dependencies(clBLAS::clBLAS clBLAS-ext)
