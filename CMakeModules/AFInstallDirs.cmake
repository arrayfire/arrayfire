#
# Sets ArrayFire installation paths.
#

# NOTE: These paths are all relative to the project installation prefix.

# Executables
if(NOT DEFINED AF_INSTALL_BIN_DIR)
  set(AF_INSTALL_BIN_DIR "bin" CACHE PATH "Installation path for executables")
endif()

# Libraries
if(NOT DEFINED AF_INSTALL_LIB_DIR)
  set(AF_INSTALL_LIB_DIR "lib" CACHE PATH "Installation path for libraries")
endif()

# Header files
if(NOT DEFINED AF_INSTALL_INC_DIR)
  set(AF_INSTALL_INC_DIR "include" CACHE PATH "Installation path for headers")
endif()

# Data files
if(NOT DEFINED AF_INSTALL_DATA_DIR)
  set(AF_INSTALL_DATA_DIR "share/ArrayFire" CACHE PATH "Installation path for data files")
endif()

# Documentation
if(NOT DEFINED AF_INSTALL_DOC_DIR)
  set(AF_INSTALL_DOC_DIR "${AF_INSTALL_DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
endif()

if(NOT DEFINED AF_INSTALL_EXAMPLE_DIR)
  set(AF_INSTALL_EXAMPLE_DIR "${AF_INSTALL_DATA_DIR}" CACHE PATH "Installation path for examples")
endif()

# Man pages
if(NOT DEFINED AF_INSTALL_MAN_DIR)
  set(AF_INSTALL_MAN_DIR "${AF_INSTALL_DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED AF_INSTALL_CMAKE_DIR)
  set(AF_INSTALL_CMAKE_DIR "${AF_INSTALL_DATA_DIR}/cmake" CACHE PATH "Installation path for CMake files")
endif()
