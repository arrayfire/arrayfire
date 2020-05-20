# picked up original content from https://crascit.com/2016/04/09/using-ccache-with-cmake/

if (UNIX)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    # Set up wrapper scripts
    set(C_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CXX_LAUNCHER "${CCACHE_PROGRAM}")
    configure_file(${PROJECT_SOURCE_DIR}/scripts/launch-c.in   launch-c)
    configure_file(${PROJECT_SOURCE_DIR}/scripts/launch-cxx.in launch-cxx)
    execute_process(COMMAND chmod a+rx
        "${PROJECT_BINARY_DIR}/launch-c"
        "${PROJECT_BINARY_DIR}/launch-cxx"
      )
    if(CMAKE_GENERATOR STREQUAL "Xcode")
      # Set Xcode project attributes to route compilation and linking
      # through our scripts
      set(CMAKE_XCODE_ATTRIBUTE_CC         "${PROJECT_BINARY_DIR}/launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_CXX        "${PROJECT_BINARY_DIR}/launch-cxx")
      set(CMAKE_XCODE_ATTRIBUTE_LD         "${PROJECT_BINARY_DIR}/launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${PROJECT_BINARY_DIR}/launch-cxx")
    else()
      # Support Unix Makefiles and Ninja
      set(CMAKE_C_COMPILER_LAUNCHER   "${PROJECT_BINARY_DIR}/launch-c")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${PROJECT_BINARY_DIR}/launch-cxx")
    endif()
  endif()
  mark_as_advanced(CCACHE_PROGRAM)
endif()
