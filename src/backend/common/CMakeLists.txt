# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

add_library(afcommon_interface INTERFACE)

target_sources(afcommon_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/BinaryNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/BinaryNode.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/ModdimNode.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/NaryNode.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/Node.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/NodeIO.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/NodeIterator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/ScalarNode.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/UnaryNode.hpp
    )

target_sources(afcommon_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/AllocatorInterface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayFireTypesIO.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DefaultMemoryManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DefaultMemoryManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FFTPlanCache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HandleBase.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/InteropManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelInterface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MemoryManagerBase.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MersenneTwister.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Source.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TemplateArg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TemplateTypename.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cast.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cblas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/complex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deterministicHash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deterministicHash.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dim4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/err_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/err_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/half.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/half.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal_enums.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_cache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moddims.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moddims.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/module_loading.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/traits.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unique_handle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp
  )

if(WIN32)
  target_sources(afcommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_windows.cpp)
else()
  target_sources(afcommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_unix.cpp)
endif()

target_link_libraries(afcommon_interface
  INTERFACE
    af_spdlog
    Boost::boost
    nonstd::span-lite
    ${CMAKE_DL_LIBS}
)

if(TARGET fmt::fmt)
  target_link_libraries(afcommon_interface
    INTERFACE
      fmt::fmt
  )
endif()

if(TARGET glad::glad)
  target_link_libraries(afcommon_interface INTERFACE glad::glad)
else()
  target_link_libraries(afcommon_interface INTERFACE af_glad)
endif()

if(AF_BUILD_FORGE AND NOT Forge_FOUND)
  add_dependencies(afcommon_interface forge)
endif()

target_include_directories(afcommon_interface
  SYSTEM INTERFACE
    $<$<PLATFORM_ID:Darwin>:${OPENGL_INCLUDE_DIR}>)

target_include_directories(afcommon_interface
  INTERFACE
    ${ArrayFire_SOURCE_DIR}/src/backend
    ${ArrayFire_BINARY_DIR}/src/backend)

if(TARGET Forge::forge)
  target_include_directories(afcommon_interface
    SYSTEM INTERFACE
      $<TARGET_PROPERTY:Forge::forge,INCLUDE_DIRECTORIES>
  )
else()
  target_include_directories(afcommon_interface
    SYSTEM INTERFACE
      ${${forge_prefix}_SOURCE_DIR}/include
      ${${forge_prefix}_BINARY_DIR}/include
  )
endif()

if(APPLE AND NOT USE_MKL)
  target_sources(afcommon_interface
    INTERFACE
      ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.hpp)
endif()
