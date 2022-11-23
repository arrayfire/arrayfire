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

add_library(opencl_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
    add_library(opencl_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp")
    add_dependencies(opencl_sort_by_key_${SBK_TYPE}
                        ${cl_kernel_targets} OpenCL::cl2hpp Boost::boost)

    target_include_directories(opencl_sort_by_key_${SBK_TYPE}
      SYSTEM PRIVATE
        ${span-lite_SOURCE_DIR}/include
        $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:OpenCL::cl2hpp,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:af_spdlog,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(opencl_sort_by_key_${SBK_TYPE}
      PRIVATE
        .
        ..
        ../../api/c
        ../common
        ../../../include
        magma
        ${ArrayFire_BINARY_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR})

    if(TARGET Forge::forge)
      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:Forge::forge,INCLUDE_DIRECTORIES>
      )
    else()
      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          ${${forge_prefix}_SOURCE_DIR}/include
          ${${forge_prefix}_BINARY_DIR}/include
      )
    endif()
    if(TARGET glad::glad)
      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:glad::glad,INTERFACE_INCLUDE_DIRECTORIES>
      )
    else()
      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:af_glad,INTERFACE_INCLUDE_DIRECTORIES>
      )
    endif()

    set_target_properties(opencl_sort_by_key_${SBK_TYPE}
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Generated Targets")

    arrayfire_set_default_cxx_flags(opencl_sort_by_key_${SBK_TYPE})

    target_compile_definitions(opencl_sort_by_key_${SBK_TYPE}
      PRIVATE
        ${opencl_compile_definitions}
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>
        TYPE=${SBK_TYPE} AFDLL)
    target_sources(opencl_sort_by_key
      INTERFACE $<TARGET_OBJECTS:opencl_sort_by_key_${SBK_TYPE}>)
endforeach(SBK_TYPE ${SBK_TYPES})
