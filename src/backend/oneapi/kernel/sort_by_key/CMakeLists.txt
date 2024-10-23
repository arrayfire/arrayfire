#Copyright(c) 2017, ArrayFire
#All rights reserved.
#
#This file is distributed under 3 - clause BSD license.
#The complete license agreement can be obtained at:
#http:  // arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_TYPES")
        string(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_TYPES ${TEMP})
    endif()
endforeach()

add_library(oneapi_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
  add_library(oneapi_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp"
    )

  if (WIN32)
    target_link_libraries(oneapi_sort_by_key_${SBK_TYPE} PRIVATE oneDPL)
  else()
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
      PROPERTIES
        LANGUAGE SYCL)
  endif()
  set_target_properties(oneapi_sort_by_key_${SBK_TYPE}
    PROPERTIES
      COMPILE_DEFINITIONS "TYPE=${SBK_TYPE};AFDLL;$<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>"
      CXX_STANDARD 17
      CXX_EXTENSIONS OFF
      CXX_VISIBILITY_PRESET hidden
      FOLDER "Generated Targets")

  arrayfire_set_default_cxx_flags(oneapi_sort_by_key_${SBK_TYPE})

  target_link_libraries(oneapi_sort_by_key_${SBK_TYPE} PRIVATE oneDPL)
  target_include_directories(oneapi_sort_by_key_${SBK_TYPE}
    PUBLIC
      .
      ../../api/c
      ${ArrayFire_SOURCE_DIR}/include
      ${ArrayFire_BINARY_DIR}/include
    PRIVATE
      ../common
      ..
      )

  target_compile_options(oneapi_sort_by_key_${SBK_TYPE}
    PRIVATE
      $<$<COMPILE_LANGUAGE:SYCL>: -fno-sycl-id-queries-fit-in-int
                                  -sycl-std=2020
                                  $<$<PLATFORM_ID:Linux>: -fno-sycl-rdc>>)

  target_include_directories(oneapi_sort_by_key_${SBK_TYPE}
    SYSTEM PRIVATE
      ${span-lite_SOURCE_DIR}/include
      $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
      ${SYCL_INCLUDE_DIR}/sycl)

  set_target_properties(oneapi_sort_by_key_${SBK_TYPE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_sources(oneapi_sort_by_key
    INTERFACE $<TARGET_OBJECTS:oneapi_sort_by_key_${SBK_TYPE}>)
endforeach(SBK_TYPE ${SBK_TYPES})
