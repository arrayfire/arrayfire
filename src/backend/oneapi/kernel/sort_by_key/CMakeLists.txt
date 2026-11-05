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

# The SYCL device linker groups kernels by source file, and the JIT compiles a
# whole group the first time any kernel in it runs. Compiling every key type
# from the same file put over 5000 kernels into one group, so the first sort
# with indices spent minutes in the GPU compiler. A generated file per key
# type, with the uint values used by sort with indices split out, keeps the
# groups small. One object per type pair would be smaller still, but the
# Windows link line then exceeds the linker's limit.
add_library(oneapi_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
  foreach(SBK_VALUES uint other)
    set(sbk_target oneapi_sort_by_key_${SBK_TYPE}_${SBK_VALUES})
    set(sbk_source
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/sort_by_key/sort_by_key_${SBK_TYPE}_${SBK_VALUES}.cpp")
    file(CONFIGURE OUTPUT "${sbk_source}"
      CONTENT "#include \"${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp\"\n")
    if(SBK_VALUES STREQUAL "uint")
      set(sbk_uint_values 1)
    else()
      set(sbk_uint_values 0)
    endif()

    add_library(${sbk_target} OBJECT
          "${sbk_source}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp"
      )

    set_source_files_properties("${sbk_source}"
      PROPERTIES
        LANGUAGE SYCL)
    set_target_properties(${sbk_target}
      PROPERTIES
        COMPILE_DEFINITIONS "TYPE=${SBK_TYPE};SBK_UINT_VALUES=${sbk_uint_values};AFDLL;$<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>"
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        FOLDER "Generated Targets")

    arrayfire_set_default_cxx_flags(${sbk_target})

    target_include_directories(${sbk_target}
      PUBLIC
        .
        ../../api/c
        ${ArrayFire_SOURCE_DIR}/include
        ${ArrayFire_BINARY_DIR}/include
      PRIVATE
        ../common
        ..
        )

    target_compile_options(${sbk_target}
      PRIVATE
        $<$<COMPILE_LANGUAGE:SYCL>: -fno-sycl-id-queries-fit-in-int
                                    -sycl-std=2020
                                    ${MSVC_RUNTIME}
                                    ${sycl_no_device_debug}
                                    $<$<PLATFORM_ID:Linux>: -fno-sycl-rdc>>)

    target_include_directories(${sbk_target}
      SYSTEM PRIVATE
        ${span-lite_SOURCE_DIR}/include
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>)

    set_target_properties(${sbk_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(oneapi_sort_by_key
      INTERFACE $<TARGET_OBJECTS:${sbk_target}>)
  endforeach()
endforeach(SBK_TYPE ${SBK_TYPES})
