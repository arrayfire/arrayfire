# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(DOWNLOAD
  "https://github.com/arrayfire/arrayfire/blob/v3.0.0/CMakeLists.txt"
  "${ArrayFire_BINARY_DIR}/download_copy_cmakelists.stamp"
  STATUS af_check_result
  TIMEOUT 4
)
list(GET af_check_result 0 af_is_connected)
if(${af_is_connected})
  set(BUILD_OFFLINE ON)
  # Turn ON disconnected flag when connected to cloud
  set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
      "Disable Download/Update stages of FetchContent workflow" FORCE)

  message(STATUS "No cloud connection. Attempting offline build if dependencies are available")
else()
  set(BUILD_OFFLINE OFF)
  # Turn OFF disconnected flag when connected to cloud
  # This is required especially in the following scenario:
  # - cmake run successfully first
  # - lost connection, but development can still be done
  # - Now, connection regained. Hence updates should be allowed
  set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL
      "Disable Download/Update stages of FetchContent workflow" FORCE)
endif()

# Track dependencies download persistently across multiple
# cmake configure runs. *_POPULATED variables are reset for each
# cmake run to 0. Hence, this internal cache value is needed to
# check for already (from previous cmake run's) populated data
# during the current cmake run if it looses network connection.
set(AF_INTERNAL_DOWNLOAD_FLAG OFF CACHE BOOL "Deps Download Flag")

# Override fetch content base dir before including AFfetch_content
set(FETCHCONTENT_BASE_DIR "${ArrayFire_BINARY_DIR}/extern" CACHE PATH
    "Base directory where ArrayFire dependencies are downloaded and/or built" FORCE)

include(AFfetch_content)

mark_as_advanced(
  AF_INTERNAL_DOWNLOAD_FLAG
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

macro(set_and_mark_depnames_advncd var name)
  string(TOLOWER ${name} ${var})
  string(TOUPPER ${name} ${var}_ucname)
  mark_as_advanced(
      FETCHCONTENT_SOURCE_DIR_${${var}_ucname}
      FETCHCONTENT_UPDATES_DISCONNECTED_${${var}_ucname}
  )
endmacro()

set_and_mark_depnames_advncd(assets_prefix "af_assets")
set_and_mark_depnames_advncd(testdata_prefix "af_test_data")
set_and_mark_depnames_advncd(gtest_prefix "googletest")
set_and_mark_depnames_advncd(glad_prefix "af_glad")
set_and_mark_depnames_advncd(forge_prefix "af_forge")
set_and_mark_depnames_advncd(spdlog_prefix "spdlog")
set_and_mark_depnames_advncd(threads_prefix "af_threads")
set_and_mark_depnames_advncd(cub_prefix "nv_cub")
set_and_mark_depnames_advncd(cl2hpp_prefix "ocl_cl2hpp")
set_and_mark_depnames_advncd(clblast_prefix "ocl_clblast")
set_and_mark_depnames_advncd(clfft_prefix "ocl_clfft")
set_and_mark_depnames_advncd(boost_prefix "boost_compute")

macro(af_dep_check_and_populate dep_prefix)
  set(single_args URI REF)
  cmake_parse_arguments(adcp_args "" "${single_args}" "" ${ARGN})

  if("${adcp_args_URI}" STREQUAL "")
    message(FATAL_ERROR [=[
        Cannot check requested dependency source's availability.
        Please provide a valid URI(almost always a URL to a github repo).
        Note that the above error message if for developers of ArrayFire.
        ]=])
  endif()

  string(FIND "${adcp_args_REF}" "=" adcp_has_algo_id)

  if(${BUILD_OFFLINE} AND NOT ${AF_INTERNAL_DOWNLOAD_FLAG})
    if(NOT ${adcp_has_algo_id} EQUAL -1)
      FetchContent_Populate(${dep_prefix}
        QUIET
        URL            ${adcp_args_URI}
        URL_HASH       ${adcp_args_REF}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${ArrayFire_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    elseif("${adcp_args_REF}" STREQUAL "")
      FetchContent_Populate(${dep_prefix}
        QUIET
        URL            ${adcp_args_URI}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${ArrayFire_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    else()
      # The left over alternative is assumed to be a cloud hosted git repository
      FetchContent_Populate(${dep_prefix}
        QUIET
        GIT_REPOSITORY ${adcp_args_URI}
        GIT_TAG        ${adcp_args_REF}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${ArrayFire_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${ArrayFire_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    endif()
  else()
    if(NOT ${adcp_has_algo_id} EQUAL -1)
      FetchContent_Declare(${dep_prefix}
        URL            ${adcp_args_URI}
        URL_HASH       ${adcp_args_REF}
      )
    elseif("${adcp_args_REF}" STREQUAL "")
      FetchContent_Declare(${dep_prefix}
        URL            ${adcp_args_URI}
      )
    else()
      # The left over alternative is assumed to be a cloud hosted git repository
      FetchContent_Declare(${dep_prefix}
        GIT_REPOSITORY ${adcp_args_URI}
        GIT_TAG        ${adcp_args_REF}
      )
    endif()
    FetchContent_GetProperties(${dep_prefix})
    if(NOT ${dep_prefix}_POPULATED)
      FetchContent_Populate(${dep_prefix})
    endif()
    set(AF_INTERNAL_DOWNLOAD_FLAG ON CACHE BOOL "Deps Download Flag" FORCE)
  endif()
endmacro()
