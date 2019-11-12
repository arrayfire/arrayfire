# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

find_program(GIT git)

set(prefix ${PROJECT_BINARY_DIR}/third_party/CLBlast)
set(CLBlast_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clblast${CMAKE_STATIC_LIBRARY_SUFFIX})

if(APPLE)
  # We need this patch on macOS until #PR 356 is merged in the CLBlast repo
  write_file(clblast.patch
"diff --git a/src/clpp11.hpp b/src/clpp11.hpp
index 9446499..786f7db 100644
--- a/src/clpp11.hpp
+++ b/src/clpp11.hpp
@@ -358,8 +358,10 @@ class Device {

   // Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)
   bool IsPostNVIDIAVolta() const {
-    assert(HasExtension(\"cl_nv_device_attribute_query\"));
-    return GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV) >= 7;
+    if(HasExtension(\"cl_nv_device_attribute_query\")) {
+      return GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV) >= 7;
+    }
+    return false;
   }

   // Retrieves the above extra information (if present)
")

  set(CLBLAST_PATCH_COMMAND ${GIT} apply ${ArrayFire_BINARY_DIR}/clblast.patch)
endif()

if(WIN32)
  set(extproj_gen_opts "-G${CMAKE_GENERATOR}" "-A${CMAKE_GENERATOR_PLATFORM}")
else(WIN32)
  set(extproj_gen_opts "-G${CMAKE_GENERATOR}")
endif(WIN32)

ExternalProject_Add(
    CLBlast-ext
    GIT_REPOSITORY https://github.com/cnugteren/CLBlast.git
    GIT_TAG 1.5.0
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CLBLAST_PATCH_COMMAND}
    BUILD_BYPRODUCTS ${CLBlast_location}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ${extproj_gen_opts}
      -Wno-dev <SOURCE_DIR>
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
      -DOVERRIDE_MSVC_FLAGS_TO_MT:BOOL=OFF
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_LIBDIR:PATH=lib
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
