# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

find_package(Boost)

set(Boost_MIN_VER 107000)
set(Boost_MIN_VER_STR "1.70")

if(NOT
  ((Boost_VERSION VERSION_GREATER Boost_MIN_VER OR
    Boost_VERSION VERSION_EQUAL Boost_MIN_VER) OR
   (Boost_VERSION_STRING VERSION_GREATER Boost_MIN_VER_STR OR
    Boost_VERSION_STRING VERSION_EQUAL Boost_MIN_VER_STR) OR
   (Boost_VERSION_MACRO VERSION_GREATER Boost_MIN_VER OR
    Boost_VERSION_MACRO VERSION_EQUAL Boost_MIN_VER)))
  set(VER 1.70.0)
  set(MD5 e160ec0ff825fc2850ea4614323b1fb5)
  include(ExternalProject)

  ExternalProject_Add(
    boost_compute
    URL       https://github.com/boostorg/compute/archive/boost-${VER}.tar.gz
    URL_MD5   ${MD5}
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    )

  ExternalProject_Get_Property(boost_compute source_dir)

  if(NOT EXISTS ${source_dir}/include)
      message(WARNING "WARN: Found Boost v${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}."
                      " Required ${VER}. Build will download Boost Compute.")
  endif()
  make_directory(${source_dir}/include)

  if(NOT TARGET Boost::boost)
    add_library(Boost::boost IMPORTED INTERFACE GLOBAL)
  endif()

  add_dependencies(Boost::boost boost_compute)

  set_target_properties(Boost::boost PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR};${source_dir}/include"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR};${source_dir}/include"
    )
endif()

if(TARGET Boost::boost)
  set(BOOST_DEFINITIONS "BOOST_CHRONO_HEADER_ONLY;BOOST_COMPUTE_THREAD_SAFE;BOOST_COMPUTE_HAVE_THREAD_LOCAL")

  # NOTE: Basic and Windows options do not requre flags or libraries for
  #       backtraces
  if(AF_STACKTRACE_TYPE STREQUAL "libbacktrace")
    list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_BACKTRACE")
    set_target_properties(Boost::boost PROPERTIES
      INTERFACE_LINK_LIBRARIES ${Backtrace_LIBRARY})
  elseif(AF_STACKTRACE_TYPE STREQUAL "addr2line")
    list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_ADDR2LINE")
  elseif(AF_STACKTRACE_TYPE STREQUAL "None")
      list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_NOOP")
  endif()

  if(NOT AF_STACKTRACE_TYPE STREQUAL "None" AND APPLE)
      list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED")
  endif()

  # NOTE: BOOST_CHRONO_HEADER_ONLY is required for Windows because otherwise it
  # will try to link with libboost-chrono.
  set_target_properties(Boost::boost PROPERTIES INTERFACE_COMPILE_DEFINITIONS
      "${BOOST_DEFINITIONS}")
endif()
