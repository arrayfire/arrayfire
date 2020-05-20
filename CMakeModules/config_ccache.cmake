# picked up original content from https://crascit.com/2016/04/09/using-ccache-with-cmake/

if (UNIX)
  find_program(CCACHE_PROGRAM ccache)

  set(CCACHE_FOUND OFF)
  if(CCACHE_PROGRAM)
    set(CCACHE_FOUND ON)
  endif()

  option(AF_USE_CCACHE "Build ArrayFire with a CPU backend" ${CCACHE_FOUND})

  if(${AF_USE_CCACHE})
    # Set up wrapper scripts
    set(C_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CXX_LAUNCHER "${CCACHE_PROGRAM}")
    configure_file(${ArrayFire_SOURCE_DIR}/CMakeModules/launch-c.in   launch-c)
    configure_file(${ArrayFire_SOURCE_DIR}/CMakeModules/launch-cxx.in launch-cxx)
    execute_process(COMMAND chmod a+rx
        "${ArrayFire_BINARY_DIR}/launch-c"
        "${ArrayFire_BINARY_DIR}/launch-cxx"
      )
    if(CMAKE_GENERATOR STREQUAL "Xcode")
      # Set Xcode project attributes to route compilation and linking
      # through our scripts
      set(CMAKE_XCODE_ATTRIBUTE_CC         "${ArrayFire_BINARY_DIR}/launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_CXX        "${ArrayFire_BINARY_DIR}/launch-cxx")
      set(CMAKE_XCODE_ATTRIBUTE_LD         "${ArrayFire_BINARY_DIR}/launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${ArrayFire_BINARY_DIR}/launch-cxx")
    else()
      # Support Unix Makefiles and Ninja
      set(CMAKE_C_COMPILER_LAUNCHER   "${ArrayFire_BINARY_DIR}/launch-c")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${ArrayFire_BINARY_DIR}/launch-cxx")
    endif()
  endif()
  mark_as_advanced(CCACHE_PROGRAM)
  mark_as_advanced(AF_USE_CCACHE)
endif()
