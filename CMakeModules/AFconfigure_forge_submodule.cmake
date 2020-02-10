# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

if(AF_BUILD_FORGE)
  set(ArrayFireInstallPrefix ${CMAKE_INSTALL_PREFIX})
  set(ArrayFireBuildType ${CMAKE_BUILD_TYPE})
  set(CMAKE_INSTALL_PREFIX ${ArrayFire_BINARY_DIR}/extern/forge/package)
  set(CMAKE_BUILD_TYPE Release)
  set(FG_BUILD_EXAMPLES OFF CACHE BOOL "Used to build Forge examples")
  set(FG_BUILD_DOCS OFF CACHE BOOL "Used to build Forge documentation")
  set(FG_WITH_FREEIMAGE OFF CACHE BOOL "Turn on usage of freeimage dependency")

  add_subdirectory(extern/forge EXCLUDE_FROM_ALL)

  mark_as_advanced(
      FG_BUILD_EXAMPLES
      FG_BUILD_DOCS
      FG_WITH_FREEIMAGE
      FG_USE_WINDOW_TOOLKIT
      FG_USE_SYSTEM_CL2HPP
      FG_ENABLE_HUNTER
      glfw3_DIR
      glm_DIR
      )
  set(CMAKE_BUILD_TYPE ${ArrayFireBuildType})
  set(CMAKE_INSTALL_PREFIX ${ArrayFireInstallPrefix})

  install(FILES
      $<TARGET_FILE:forge>
      $<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:forge>>
      $<$<PLATFORM_ID:Darwin>:$<TARGET_SONAME_FILE:forge>>
      DESTINATION "${AF_INSTALL_LIB_DIR}"
      COMPONENT common_backend_dependencies)
  set_property(TARGET forge APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
else(AF_BUILD_FORGE)
  set(FG_VERSION "1.0.0")
  set(FG_VERSION_MAJOR 1)
  set(FG_VERSION_MINOR 0)
  set(FG_VERSION_PATCH 0)
  set(FG_API_VERSION_CURRENT 10)
  configure_file(
    ${PROJECT_SOURCE_DIR}/extern/forge/CMakeModules/version.h.in
    ${PROJECT_BINARY_DIR}/extern/forge/include/fg/version.h
    )
endif(AF_BUILD_FORGE)
