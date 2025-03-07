# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

af_dep_check_and_populate(${clfft_prefix}
  URI https://github.com/arrayfire/clFFT.git
  REF arrayfire-release
)

set(current_build_type ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${${clfft_prefix}_SOURCE_DIR}/src ${${clfft_prefix}_BINARY_DIR} EXCLUDE_FROM_ALL)
get_property(clfft_include_dir
  TARGET clFFT
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(clFFT
  PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${clfft_include_dir}")

# OpenCL targets need this flag to avoid ignored attribute warnings in the
# OpenCL headers
check_cxx_compiler_flag(-Wno-ignored-attributes has_ignored_attributes_flag)
if(has_ignored_attributes_flag)
  target_compile_options(clFFT
    PRIVATE -Wno-ignored-attributes)
endif()
set(BUILD_SHARED_LIBS ${current_build_type})

mark_as_advanced(
  Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE
  CLFFT_BUILD64
  CLFFT_BUILD_CALLBACK_CLIENT
  CLFFT_BUILD_CLIENT
  CLFFT_BUILD_EXAMPLES
  CLFFT_BUILD_LOADLIBRARIES
  CLFFT_BUILD_RUNTIME
  CLFFT_BUILD_TEST
  CLFFT_CODE_COVERAGE
  CLFFT_SUFFIX_BIN
  CLFFT_SUFFIX_LIB
)
