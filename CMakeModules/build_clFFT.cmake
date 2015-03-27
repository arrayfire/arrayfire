include(ExternalProject)

set(prefix "${CMAKE_BINARY_DIR}/third_party/clFFT")
set(clFFT_location ${prefix}/lib/import/${CMAKE_STATIC_LIBRARY_PREFIX}clFFT${CMAKE_STATIC_LIBRARY_SUFFIX})
if(CMAKE_VERSION VERSION_LESS 3.2 AND CMAKE_GENERATOR MATCHES "Ninja")
  message(WARNING "Building clFFT with Ninja has known issues with CMake older than 3.2")
  set(byproducts)
else()
  set(byproducts BYPRODUCTS ${clFFT_location})
endif()

ExternalProject_Add(
  clFFT-external
  GIT_REPOSITORY https://github.com/arrayfire/clFFT.git
  GIT_TAG 1c14c415d1dd8d9a73fa2546086508a34f2008af
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
    -DBUILD_CLIENT:BOOL=OFF
    -DBUILD_SHARED_LIBRARY:BOOL=OFF
    -DBUILD_TEST:BOOL=OFF
    -DSUFFIX_LIB:STRING=
    -DUSE_SYSTEM_GTEST:BOOL=ON
  ${byproducts})

ExternalProject_Get_Property(clFFT-external install_dir)
add_library(clFFT IMPORTED STATIC)
set_target_properties(clFFT PROPERTIES IMPORTED_LOCATION ${clFFT_location})
add_dependencies(clFFT clFFT-external)
set(CLFFT_INCLUDE_DIRS ${install_dir}/include)
set(CLFFT_LIBRARIES clFFT)
set(CLFFT_FOUND ON)
