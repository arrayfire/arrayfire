#
# Sets ArrayFire installation paths.
#

include(GNUInstallDirs)

# NOTE: These paths are all relative to the project installation prefix.

# Executables
if(NOT DEFINED AF_INSTALL_BIN_DIR)
  set(AF_INSTALL_BIN_DIR "lib" CACHE PATH "Installation path for executables")
endif()

# Libraries
if(NOT DEFINED AF_INSTALL_LIB_DIR)
  if(WIN32)
    set(AF_INSTALL_LIB_DIR "lib" CACHE PATH "Installation path for libraries")
  else()
    set(AF_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Installation path for libraries")
  endif()
endif()

# Header files
if(NOT DEFINED AF_INSTALL_INC_DIR)
  set(AF_INSTALL_INC_DIR "include" CACHE PATH "Installation path for headers")
endif()

set(DATA_DIR "share/ArrayFire")

# Documentation
if(NOT DEFINED AF_INSTALL_DOC_DIR)
  if (WIN32)
    set(AF_INSTALL_DOC_DIR "doc" CACHE PATH "Installation path for documentation")
  else ()
      set(AF_INSTALL_DOC_DIR "${DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
  endif ()
endif()

if(NOT DEFINED AF_INSTALL_EXAMPLE_DIR)
  if (WIN32)
    set(AF_INSTALL_EXAMPLE_DIR "examples" CACHE PATH "Installation path for examples")
  else ()
    set(AF_INSTALL_EXAMPLE_DIR "${DATA_DIR}/examples" CACHE PATH "Installation path for examples")
  endif ()
endif()

# Man pages
if(NOT DEFINED AF_INSTALL_MAN_DIR)
    set(AF_INSTALL_MAN_DIR "${DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED AF_INSTALL_CMAKE_DIR)
  if (WIN32)
    set(AF_INSTALL_CMAKE_DIR "cmake" CACHE PATH "Installation path for CMake files")
  else ()
    set(AF_INSTALL_CMAKE_DIR "${DATA_DIR}/cmake" CACHE PATH "Installation path for CMake files")
  endif ()
endif()

mark_as_advanced(
  AF_INSTALL_BIN_DIR
  AF_INSTALL_LIB_DIR
  AF_INSTALL_INC_DIR
  AF_INSTALL_DATA_DIR
  AF_INSTALL_DOC_DIR
  AF_INSTALL_EXAMPLE_DIR
  AF_INSTALL_MAN_DIR
  AF_INSTALL_CMAKE_DIR)
