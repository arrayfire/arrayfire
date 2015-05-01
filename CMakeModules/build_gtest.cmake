# Build the gtest libraries

# Check if Google Test exists
SET(GTEST_SOURCE_DIR "${CMAKE_SOURCE_DIR}/test/gtest")
IF(NOT EXISTS "${GTEST_SOURCE_DIR}/README")
    MESSAGE(WARNING "GTest Source is not available. Tests will not build.")
    MESSAGE("Did you miss the --recursive option when cloning?")
    MESSAGE("Run the following commands to correct this:")
    MESSAGE("git submodule init")
    MESSAGE("git submodule update")
    MESSAGE("git submodule foreach git pull origin master")
ENDIF()

if(CMAKE_VERSION VERSION_LESS 3.2 AND CMAKE_GENERATOR MATCHES "Ninja")
    message(WARNING "Building GTest with Ninja has known issues with CMake older than 3.2")
endif()

include(ExternalProject)

# Set the build type if it isn't already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set default ExternalProject root directory
set(prefix "${CMAKE_BINARY_DIR}/third_party/gtest")
# the binary dir must be know before creating the external project in order
# to pass the byproducts
set(binary_dir "${prefix}/src/googletest-build")
set(stdlib_binary_dir "${prefix}/src/googletest-build-stdlib")

set(GTEST_LIBRARIES gtest gtest_main)
set(GTEST_LIBRARIES_STDLIB gtest_stdlib gtest_main_stdlib)

set(byproducts)
set(byproducts_libstdcpp)
foreach(lib ${GTEST_LIBRARIES})
    set(${lib}_location
        ${binary_dir}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(${lib}_location_libstdcpp
        ${stdlib_binary_dir}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX})
    list(APPEND byproducts ${${lib}_location})
    list(APPEND byproducts_libstdcpp ${${lib}_location_libstdcpp})
endforeach()
SET(CMAKE_CXX_FLAGS_STD "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

FUNCTION(GTEST_BUILD BUILD_NAME BUILD_TYPE BUILD_BINARY_DIR BUILD_BYPRODUCTS)
# Add gtest
ExternalProject_Add(
    ${BUILD_NAME}
    # URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
    # URL_MD5 2d6ec8ccdf5c46b05ba54a9fd1d130d7
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test/gtest"
    PREFIX ${prefix}
    BINARY_DIR ${BUILD_BINARY_DIR}
    TIMEOUT 10
    CMAKE_ARGS -Dgtest_force_shared_crt=ON
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_LIBSTDCPP=${CMAKE_CXX_FLAGS_STD}
               -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
               -DCMAKE_CXX_FLAGS_MINSIZEREL=${CMAKE_CXX_FLAGS_MINSIZEREL}
               -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS}
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD 0
    LOG_UPDATE 0
    LOG_CONFIGURE 0
    LOG_BUILD 0)
ENDFUNCTION(GTEST_BUILD)

GTEST_BUILD(googletest              ${CMAKE_BUILD_TYPE} ${binary_dir} "${byproducts}")

# If we are on OSX and using the clang compiler go ahead and build
# GTest using libstdc++ just in case we compile the CUDA backend
IF("${APPLE}" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    GTEST_BUILD(googletest_libstdcpp    LibStdCpp           ${stdlib_binary_dir} "${byproducts_libstdcpp}")
ENDIF("${APPLE}" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")

foreach(lib ${GTEST_LIBRARIES})
    add_library(${lib} IMPORTED STATIC)
    add_dependencies(${lib} googletest)
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${${lib}_location})

    IF("${APPLE}" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        add_library(${lib}_stdlib IMPORTED STATIC)
        add_dependencies(${lib}_stdlib googletest_libstdcpp)
        set_target_properties(${lib}_stdlib PROPERTIES IMPORTED_LOCATION ${${lib}_location_libstdcpp})
    ENDIF("${APPLE}" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
endforeach()

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/include)
set(GTEST_FOUND ON)
