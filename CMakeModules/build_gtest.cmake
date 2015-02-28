# Build the gtest libraries
include(ExternalProject)

# Set the build type if it isn't already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX "${CMAKE_BINARY_DIR}/third_party")

# Add gtest
ExternalProject_Add(
    googletest
    GIT_SUBMODULES test/gtest
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test/gtest"
    TIMEOUT 10
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO:PATH=ReleaseLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD 0
    LOG_UPDATE 0
    LOG_CONFIGURE 0
    LOG_BUILD 0)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
include_directories("${source_dir}/include")
