# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(FG_VERSION_MAJOR 1)
set(FG_VERSION_MINOR 0)
set(FG_VERSION_PATCH 8)
set(FG_VERSION "${FG_VERSION_MAJOR}.${FG_VERSION_MINOR}.${FG_VERSION_PATCH}")
set(FG_API_VERSION_CURRENT ${FG_VERSION_MAJOR}${FG_VERSION_MINOR})


if(AF_BUILD_FORGE)
    af_dep_check_and_populate(${forge_prefix}
        URI https://github.com/arrayfire/forge.git
        REF "v${FG_VERSION}"
    )

    set(af_FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR})
    set(af_FETCHCONTENT_QUIET ${FETCHCONTENT_QUIET})
    set(af_FETCHCONTENT_FULLY_DISCONNECTED ${FETCHCONTENT_FULLY_DISCONNECTED})
    set(af_FETCHCONTENT_UPDATES_DISCONNECTED ${FETCHCONTENT_UPDATES_DISCONNECTED})

    set(ArrayFireInstallPrefix ${CMAKE_INSTALL_PREFIX})
    set(ArrayFireBuildType ${CMAKE_BUILD_TYPE})
    set(CMAKE_INSTALL_PREFIX ${${forge_prefix}_BINARY_DIR}/extern/forge/package)
    set(CMAKE_BUILD_TYPE Release)
    set(FG_BUILD_EXAMPLES OFF CACHE BOOL "Used to build Forge examples")
    set(FG_BUILD_DOCS OFF CACHE BOOL "Used to build Forge documentation")
    set(FG_WITH_FREEIMAGE OFF CACHE BOOL "Turn on usage of freeimage dependency")

    add_subdirectory(
        ${${forge_prefix}_SOURCE_DIR} ${${forge_prefix}_BINARY_DIR} EXCLUDE_FROM_ALL)
    mark_as_advanced(
        FG_BUILD_EXAMPLES
        FG_BUILD_DOCS
        FG_WITH_FREEIMAGE
        FG_USE_WINDOW_TOOLKIT
        FG_RENDERING_BACKEND
        SPHINX_EXECUTABLE
        glfw3_DIR
        glm_DIR
        )
    set(CMAKE_BUILD_TYPE ${ArrayFireBuildType})
    set(CMAKE_INSTALL_PREFIX ${ArrayFireInstallPrefix})
    set(FETCHCONTENT_BASE_DIR ${af_FETCHCONTENT_BASE_DIR})
    set(FETCHCONTENT_QUIET ${af_FETCHCONTENT_QUIET})
    set(FETCHCONTENT_FULLY_DISCONNECTED ${af_FETCHCONTENT_FULLY_DISCONNECTED})
    set(FETCHCONTENT_UPDATES_DISCONNECTED ${af_FETCHCONTENT_UPDATES_DISCONNECTED})
    install(FILES
        $<TARGET_FILE:forge>
        $<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:forge>>
        $<$<PLATFORM_ID:Darwin>:$<TARGET_SONAME_FILE:forge>>
        $<$<PLATFORM_ID:Linux>:$<TARGET_LINKER_FILE:forge>>
        $<$<PLATFORM_ID:Darwin>:$<TARGET_LINKER_FILE:forge>>
        DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT common_backend_dependencies)

    if(AF_INSTALL_STANDALONE)
        cmake_minimum_required(VERSION 3.21)
        install(FILES
            $<TARGET_RUNTIME_DLLS:forge>
            DESTINATION "${AF_INSTALL_LIB_DIR}"
            COMPONENT common_backend_dependencies)
    endif(AF_INSTALL_STANDALONE)

    set_property(TARGET forge APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
else(AF_BUILD_FORGE)
    find_package(Forge
        ${FG_VERSION_MAJOR}.${FG_VERSION_MINOR}.${FG_VERSION_PATCH}
        QUIET
    )

    if(TARGET Forge::forge)
        get_target_property(fg_lib_type Forge::forge TYPE)
        if(NOT ${fg_lib_type} STREQUAL "STATIC_LIBRARY" AND
           AF_INSTALL_STANDALONE)
            install(FILES
                    $<TARGET_FILE:Forge::forge>
                    $<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:Forge::forge>>
                    $<$<PLATFORM_ID:Darwin>:$<TARGET_SONAME_FILE:Forge::forge>>
                    $<$<PLATFORM_ID:Linux>:$<TARGET_LINKER_FILE:Forge::forge>>
                    $<$<PLATFORM_ID:Darwin>:$<TARGET_LINKER_FILE:Forge::forge>>
                    DESTINATION "${AF_INSTALL_LIB_DIR}"
                    COMPONENT common_backend_dependencies)
        endif()
    else()
        af_dep_check_and_populate(${forge_prefix}
            URI https://github.com/arrayfire/forge.git
            REF "v${FG_VERSION}"
        )

        configure_file(
            ${${forge_prefix}_SOURCE_DIR}/CMakeModules/version.h.in
            ${${forge_prefix}_BINARY_DIR}/include/fg/version.h
        )
    endif()
endif(AF_BUILD_FORGE)
