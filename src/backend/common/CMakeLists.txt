# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

add_library(afcommon_interface INTERFACE)

target_sources(afcommon_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FFTPlanCache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MatrixAlgebraHandle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MemoryManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MersenneTwister.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cblas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dim4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/err_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/err_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/module_loading.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp
    ${PROJECT_BINARY_DIR}/version.hpp
  )

if(WIN32)
  target_sources(afcommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_windows.cpp)
else()
  target_sources(afcommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_unix.cpp)
endif()

target_include_directories(afcommon_interface
  INTERFACE
    ${CMAKE_SOURCE_DIR}/src/backend
    ${PROJECT_BINARY_DIR}
  )

if(AF_WITH_LOGGING)
  dependency_check(spdlog_FOUND "spdlog not found.")
  target_compile_definitions(afcommon_interface
    INTERFACE AF_WITH_LOGGING)
  target_link_libraries(afcommon_interface
    INTERFACE
      spdlog::spdlog)
endif()

if(APPLE AND NOT USE_MKL)
  target_sources(afcommon_interface
    INTERFACE
      ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.hpp)
endif()

if(AF_WITH_GRAPHICS)
  dependency_check(glbinding_FOUND "glbinding not found.")

  target_include_directories(afcommon_interface
    INTERFACE
      ${Forge_INCLUDE_DIR}
    )
  target_sources(afcommon_interface
    INTERFACE
      ${CMAKE_CURRENT_SOURCE_DIR}/InteropManager.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/InteropManager.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.hpp
      )

  target_link_libraries(afcommon_interface
    INTERFACE
      OpenGL::GL
      ${Forge_LIBRARIES})

  target_compile_definitions(afcommon_interface INTERFACE WITH_GRAPHICS)

  if(APPLE)
    # TODO: On APPLE platform linking directly against glbinding brings in flags
    # that causes issues when building ArrayFire with LAPACK and Graphics. This
    # was due to the way glbinding was brining in some Framework flags which
    # cause issues with the Accelerate Framework. This is probably a bug in
    # glbindings cmake file
    target_link_libraries(afcommon_interface
      INTERFACE
        $<TARGET_PROPERTY:glbinding::glbinding,IMPORTED_LOCATION_RELEASE>)
  else()
    target_link_libraries(afcommon_interface INTERFACE glbinding::glbinding)
  endif()

  if(AF_INSTALL_STANDALONE)
    install(FILES
        $<TARGET_FILE:glbinding::glbinding>
        $<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:glbinding::glbinding>>
        $<$<PLATFORM_ID:Darwin>:$<TARGET_SONAME_FILE:glbinding::glbinding>>
      DESTINATION ${AF_INSTALL_LIB_DIR}
      COMPONENT common_backend_dependencies)
  endif()

  if(NOT AF_USE_SYSTEM_FORGE)
    add_dependencies(afcommon_interface forge-ext)
  endif()
endif()
