include(ExternalProject)

set(prefix ${CMAKE_BINARY_DIR}/third_party/clBLAS)
set(clBLAS_location ${prefix}/lib/import/${CMAKE_STATIC_LIBRARY_PREFIX}clBLAS${CMAKE_STATIC_LIBRARY_SUFFIX})
if(CMAKE_VERSION VERSION_LESS 3.2 AND CMAKE_GENERATOR MATCHES "Ninja")
  message(WARNING "Building clBLAS with Ninja has known issues with CMake older than 3.2")
  set(byproducts)
else()
  set(byproducts BYPRODUCTS ${clBLAS_location})
endif()

ExternalProject_Add(
  clBLAS-external
  GIT_REPOSITORY https://github.com/arrayfire/clBLAS.git
  GIT_TAG 47662a6ac1186c756508109d7fef8827efab4504
  PREFIX "${prefix}"
  INSTALL_DIR "${prefix}"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>/src
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w"
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w"
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_CLIENT:BOOL=OFF
    -DBUILD_TEST:BOOL=OFF
    -DBUILD_KTEST:BOOL=OFF
    -DSUFFIX_LIB:STRING=
  ${byproducts})

ExternalProject_Get_Property(clBLAS-external install_dir)
add_library(clBLAS IMPORTED STATIC)
set_target_properties(clBLAS PROPERTIES IMPORTED_LOCATION ${clBLAS_location})
add_dependencies(clBLAS clBLAS-external)
set(CLBLAS_INCLUDE_DIRS ${install_dir}/include)
set(CLBLAS_LIBRARIES clBLAS)
set(CLBLAS_FOUND ON)
