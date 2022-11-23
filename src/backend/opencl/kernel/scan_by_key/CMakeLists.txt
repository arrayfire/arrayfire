# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp" FILESTRINGS)
find_package(OpenCL REQUIRED)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_BINARY_OPS")
        string(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    endif()
endforeach()

add_library(opencl_scan_by_key INTERFACE)

add_dependencies(opencl_scan_by_key ${cl_kernel_targets} cl2hpp)
foreach(SBK_BINARY_OP ${SBK_BINARY_OPS})
    add_library(opencl_scan_by_key_${SBK_BINARY_OP} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key_impl.hpp")

    add_dependencies(opencl_scan_by_key_${SBK_BINARY_OP}
                        ${cl_kernel_targets} OpenCL::cl2hpp Boost::boost)

    target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
      PRIVATE
        .
        ..
        magma
        ../../api/c
        ../common
        ../../../include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${ArrayFire_BINARY_DIR}/include)
    target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
      SYSTEM PRIVATE
        $<TARGET_PROPERTY:af_spdlog,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:OpenCL::cl2hpp,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:nonstd::span-lite,INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
      )
    if(TARGET Forge::forge)
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:Forge::forge,INCLUDE_DIRECTORIES>
      )
    else()
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
          ${${forge_prefix}_SOURCE_DIR}/include
          ${${forge_prefix}_BINARY_DIR}/include
      )
    endif()
    if(TARGET glad::glad)
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:glad::glad,INTERFACE_INCLUDE_DIRECTORIES>
      )
    else()
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:af_glad,INTERFACE_INCLUDE_DIRECTORIES>
      )
    endif()

    set_target_properties(opencl_scan_by_key_${SBK_BINARY_OP}
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Generated Targets")

    arrayfire_set_default_cxx_flags(opencl_scan_by_key_${SBK_BINARY_OP})
    target_compile_definitions(opencl_scan_by_key_${SBK_BINARY_OP}
      PRIVATE
        ${opencl_compile_definitions}
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:af_spdlog,INTERFACE_COMPILE_DEFINITIONS>
        TYPE=${SBK_BINARY_OP} AFDLL)
    target_sources(opencl_scan_by_key
      INTERFACE $<TARGET_OBJECTS:opencl_scan_by_key_${SBK_BINARY_OP}>)
endforeach(SBK_BINARY_OP ${SBK_BINARY_OPS})
