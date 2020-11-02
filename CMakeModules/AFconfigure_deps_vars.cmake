# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

option(AF_BUILD_OFFLINE "Build ArrayFire assuming there is no network" OFF)

# Override fetch content base dir before including AFfetch_content
set(FETCHCONTENT_BASE_DIR "${ArrayFire_BINARY_DIR}/extern" CACHE PATH
    "Base directory where ArrayFire dependencies are downloaded and/or built" FORCE)

include(AFfetch_content)

macro(set_and_mark_depname var name)
  string(TOLOWER ${name} ${var})
  string(TOUPPER ${name} ${var}_ucname)
  mark_as_advanced(
      FETCHCONTENT_SOURCE_DIR_${${var}_ucname}
      FETCHCONTENT_UPDATES_DISCONNECTED_${${var}_ucname}
  )
endmacro()

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

set_and_mark_depname(assets_prefix "af_assets")
set_and_mark_depname(testdata_prefix "af_test_data")
set_and_mark_depname(gtest_prefix "googletest")
set_and_mark_depname(glad_prefix "af_glad")
set_and_mark_depname(forge_prefix "af_forge")
set_and_mark_depname(spdlog_prefix "spdlog")
set_and_mark_depname(threads_prefix "af_threads")
set_and_mark_depname(cub_prefix "nv_cub")

if(AF_BUILD_OFFLINE)
  macro(set_fetchcontent_src_dir prefix_var dep_name)
    set(FETCHCONTENT_SOURCE_DIR_${${prefix_var}_ucname}
        "${FETCHCONTENT_BASE_DIR}/${${prefix_var}}-src" CACHE PATH
        "Source directory for ${dep_name} dependency")
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_${${prefix_var}_ucname})
  endmacro()

  set_fetchcontent_src_dir(assets_prefix "Assets")
  set_fetchcontent_src_dir(testdata_prefix "Test Data")
  set_fetchcontent_src_dir(gtest_prefix "googletest")
  set_fetchcontent_src_dir(glad_prefix "glad")
  set_fetchcontent_src_dir(forge_prefix "forge")
  set_fetchcontent_src_dir(spdlog_prefix "spdlog")
  set_fetchcontent_src_dir(threads_prefix "threads")
  set_fetchcontent_src_dir(cub_prefix "NVIDIA CUB")
endif()
