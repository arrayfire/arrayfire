MESSAGE(STATUS "Adding Fast Build Type")

IF(MSVC)

    SET(CMAKE_CXX_FLAGS_FAST
        "/MD /Od /Ob1 /D NDEBUG"
        CACHE STRING "Flags used by the C++ compiler during Fast builds."
        FORCE )
    SET(CMAKE_C_FLAGS_FAST
        "/MD /Od /Ob1 /D NDEBUG"
        CACHE STRING "Flags used by the C compiler during Fast builds."
        FORCE )
    SET(CMAKE_EXE_LINKER_FLAGS_FAST
        "/INCREMENTAL:NO"
        CACHE STRING "Flags used for linking binaries during Fast builds."
        FORCE )
    SET(CMAKE_MODULE_LINKER_FLAGS_FAST
        "/INCREMENTAL:NO"
        CACHE STRING "Flags used by the modules linker during Fast builds."
        FORCE )
    SET(CMAKE_STATIC_LINKER_FLAGS_FAST
        ""
        CACHE STRING "Flags used by the static libraries linker during Fast builds."
        FORCE )
    SET(CMAKE_SHARED_LINKER_FLAGS_FAST
        "/INCREMENTAL:NO"
        CACHE STRING "Flags used by the shared libraries linker during Fast builds."
        FORCE )

    LIST(APPEND CMAKE_CONFIGURATION_TYPES Fast)
    LIST(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
        "Semicolon separated list of supported configuration types [Debug|Release|MinSizeRel|RelWithDebInfo|Fast]"
        FORCE)

    # Needed for config to show up in VS
    # http://cmake.3232098.n2.nabble.com/Custom-configuration-types-in-Visual-Studio-td7181786.html
    ENABLE_LANGUAGE(CXX)

ELSE(MSVC)

    SET(CMAKE_CXX_FLAGS_FAST
        "-O0 -DNDEBUG"
        CACHE STRING "Flags used by the C++ compiler during Fast builds."
        FORCE )
    SET(CMAKE_C_FLAGS_FAST
        "-O0 -DNDEBUG"
        CACHE STRING "Flags used by the C compiler during Fast builds."
        FORCE )
    SET(CMAKE_EXE_LINKER_FLAGS_FAST
        ""
        CACHE STRING "Flags used for linking binaries during Fast builds."
        FORCE )
    SET(CMAKE_MODULE_LINKER_FLAGS_FAST
        ""
        CACHE STRING "Flags used by the modules linker during Fast builds."
        FORCE )
    SET(CMAKE_STATIC_LINKER_FLAGS_FAST
        ""
        CACHE STRING "Flags used by the static libraries linker during Fast builds."
        FORCE )
    SET(CMAKE_SHARED_LINKER_FLAGS_FAST
        ""
        CACHE STRING "Flags used by the shared libraries linker during Fast builds."
        FORCE )

ENDIF(MSVC)

SET(FAST_CONFIG_ENABLED ON CACHE INTERNAL "" FORCE)

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_FAST
    CMAKE_C_FLAGS_FAST
    CMAKE_EXE_LINKER_FLAGS_FAST
    CMAKE_MODULE_LINKER_FLAGS_FAST
    CMAKE_STATIC_LINKER_FLAGS_FAST
    CMAKE_SHARED_LINKER_FLAGS_FAST
    )
