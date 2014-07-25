#Downloads and installs GTest into the third_party directory

# Create patch file for gtest with MSVC 2012
if(MSVC_VERSION EQUAL 1700)
  file(WRITE ${CMAKE_BINARY_DIR}/gtest.patch "Index: cmake/internal_utils.cmake\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "===================================================================\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "--- cmake/internal_utils.cmake   (revision 660)\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "+++ cmake/internal_utils.cmake   (working copy)\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "@@ -66,6 +66,9 @@\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "       # Resolved overload was found by argument-dependent lookup.\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "       set(cxx_base_flags \"\${cxx_base_flags} -wd4675\")\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "     endif()\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "+    if (MSVC_VERSION EQUAL 1700)\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "+      set(cxx_base_flags \"\${cxx_base_flags} -D_VARIADIC_MAX=10\")\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "+    endif ()\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "     set(cxx_base_flags \"\${cxx_base_flags} -D_UNICODE -DUNICODE -DWIN32 -D_WIN32\")\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "     set(cxx_base_flags \"\${cxx_base_flags} -DSTRICT -DWIN32_LEAN_AND_MEAN\")\n")
  file(APPEND ${CMAKE_BINARY_DIR}/gtest.patch "     set(cxx_exception_flags \"-EHsc -D_HAS_EXCEPTIONS=1\")\n")
else()
  file(WRITE ${CMAKE_BINARY_DIR}/gtest.patch "")
endif()

# Enable ExternalProject CMake module
include(ExternalProject)

# Set the build type if it isn't already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

# Add gtest
ExternalProject_Add(
    googletest
    SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk/
    SVN_REVISION -r 660
    TIMEOUT 10
    PATCH_COMMAND svn patch ${CMAKE_BINARY_DIR}/gtest.patch ${CMAKE_BINARY_DIR}/third_party/src/googletest
    # Force separate output paths for debug and release builds to allow easy
    # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD 0
    LOG_UPDATE 0
    LOG_CONFIGURE 0
    LOG_BUILD 0)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
include_directories(${source_dir}/include)
