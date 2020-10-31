# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(FG_VERSION_MAJOR 1)
set(FG_VERSION_MINOR 0)
set(FG_VERSION_PATCH 5)
set(FG_VERSION "${FG_VERSION_MAJOR}.${FG_VERSION_MINOR}.${FG_VERSION_PATCH}")
set(FG_API_VERSION_CURRENT ${FG_VERSION_MAJOR}${FG_VERSION_MINOR})

FetchContent_Declare(
  af_forge
  GIT_REPOSITORY https://github.com/arrayfire/forge.git
  GIT_TAG        "v${FG_VERSION}"
)
FetchContent_Populate(af_forge)
if(AF_BUILD_FORGE)
  set(ArrayFireInstallPrefix ${CMAKE_INSTALL_PREFIX})
  set(ArrayFireBuildType ${CMAKE_BUILD_TYPE})
  set(CMAKE_INSTALL_PREFIX ${af_forge_BINARY_DIR}/extern/forge/package)
  set(CMAKE_BUILD_TYPE Release)
  set(FG_BUILD_EXAMPLES OFF CACHE BOOL "Used to build Forge examples")
  set(FG_BUILD_DOCS OFF CACHE BOOL "Used to build Forge documentation")
  set(FG_WITH_FREEIMAGE OFF CACHE BOOL "Turn on usage of freeimage dependency")

  add_subdirectory(${af_forge_SOURCE_DIR} ${af_forge_BINARY_DIR} EXCLUDE_FROM_ALL)

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
      $<$<PLATFORM_ID:Linux>:$<TARGET_LINKER_FILE:forge>>
      $<$<PLATFORM_ID:Darwin>:$<TARGET_LINKER_FILE:forge>>
      DESTINATION "${AF_INSTALL_LIB_DIR}"
      COMPONENT common_backend_dependencies)
  set_property(TARGET forge APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
else(AF_BUILD_FORGE)
  configure_file(
    ${af_forge_SOURCE_DIR}/CMakeModules/version.h.in
    ${af_forge_BINARY_DIR}/include/fg/version.h
    )
endif(AF_BUILD_FORGE)
