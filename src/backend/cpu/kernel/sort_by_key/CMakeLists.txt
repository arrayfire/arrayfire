# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_TYPES")
        string(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_TYPES ${TEMP})
    endif()
endforeach()

add_library(cpu_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
  add_library(cpu_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp"
    )
  set_target_properties(cpu_sort_by_key_${SBK_TYPE}
    PROPERTIES
      COMPILE_DEFINITIONS "TYPE=${SBK_TYPE};AFDLL;$<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>"
      FOLDER "Generated Targets")

  arrayfire_set_default_cxx_flags(cpu_sort_by_key_${SBK_TYPE})

  target_include_directories(cpu_sort_by_key_${SBK_TYPE}
    PUBLIC
      .
      ../../api/c
      ${ArrayFire_SOURCE_DIR}/include
      ${ArrayFire_BINARY_DIR}/include
    PRIVATE
      ../common
      ..
      threads)

  target_include_directories(cpu_sort_by_key_${SBK_TYPE}
    SYSTEM PRIVATE
      $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>)

  set_target_properties(cpu_sort_by_key_${SBK_TYPE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_sources(cpu_sort_by_key
    INTERFACE $<TARGET_OBJECTS:cpu_sort_by_key_${SBK_TYPE}>)
endforeach(SBK_TYPE ${SBK_TYPES})
