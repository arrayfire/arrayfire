#
# Sets ArrayFire installation paths.
#

# These paths are all relative to the project installation prefix.
set(AF_INSTALL_BIN_DIR bin CACHE PATH "Installation path for executables")
set(AF_INSTALL_LIB_DIR lib CACHE PATH "Installation path for libraries")
set(AF_INSTALL_INC_DIR include CACHE PATH "Installation path for headers")
set(AF_INSTALL_DATA_DIR share/ArrayFire CACHE PATH "Installation path for data files")
set(AF_INSTALL_DOC_DIR ${AF_INSTALL_DATA_DIR}/doc CACHE PATH "Installation path for documentation")
set(AF_INSTALL_MAN_DIR ${AF_INSTALL_DATA_DIR}/man CACHE PATH "Installation path for man pages")
set(AF_INSTALL_CMAKE_DIR ${AF_INSTALL_DATA_DIR}/cmake CACHE PATH "Installation path for CMake modules")
