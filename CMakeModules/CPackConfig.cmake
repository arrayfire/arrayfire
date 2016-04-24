CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE("${CMAKE_MODULE_PATH}/Version.cmake")

OPTION(CREATE_STGZ "Create .sh install file" ON)
MARK_AS_ADVANCED(CREATE_STGZ)

# CPack package generation
IF(${CREATE_STGZ})
  LIST(APPEND CPACK_GENERATOR "STGZ")
ENDIF()

OPTION(CREATE_DEB "Create .deb install file" OFF)
MARK_AS_ADVANCED(CREATE_DEB)

IF(${CREATE_DEB})
  LIST(APPEND CPACK_GENERATOR "DEB")
ENDIF()

OPTION(CREATE_RPM "Create .rpm install file" OFF)
MARK_AS_ADVANCED(CREATE_RPM)

IF(${CREATE_RPM})
  LIST(APPEND CPACK_GENERATOR "RPM")
ENDIF()

# Common settings to all packaging tools
SET(CPACK_PREFIX_DIR ${CMAKE_INSTALL_PREFIX})
SET(CPACK_PACKAGE_NAME "arrayfire")
SET(CPACK_PACKAGE_VERSION ${AF_VERSION})
SET(CPACK_PACKAGE_VERSION_MAJOR "${AF_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${AF_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${AF_VERSION_PATCH}")
IF(BUILD_GRAPHICS)
    SET(CPACK_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})
ELSE()
    SET(CPACK_PACKAGE_FILE_NAME
        ${CPACK_PACKAGE_NAME}-no-gl-${CPACK_PACKAGE_VERSION})
ENDIF()
SET(CPACK_PACKAGE_VENDOR "ArrayFire")
SET(CPACK_PACKAGE_CONTACT "ArrayFire Development Group <technical@arrayfire.com>")
SET(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
SET(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")

# Long description of the package
SET(CPACK_PACKAGE_DESCRIPTION
"ArrayFire is a high performance software library for parallel computing
with an easy-to-use API. Its array based function set makes parallel
programming simple.

ArrayFire's multiple backends (CUDA, OpenCL and native CPU) make it
platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel
computing code, saving you valuable time and lowering development costs.")

# Short description of the package
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A high performance library for parallel computing with an easy-to-use API.")

# Useful descriptions for components
SET(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "ArrayFire libraries")
SET(CPACK_COMPONENT_DOCUMENTATION_NAME "Doxygen documentation")
SET(CPACK_COMPONENT_HEADERS_NAME "C/C++ headers")
SET(CPACK_COMPONENT_CMAKE_NAME "CMake support")
# Set the default components installed in the package
SET(CPACK_COMPONENTS_ALL libraries headers documentation cmake)

##
# Debian package
##
SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${PROCESSOR_ARCHITECTURE})

##
# RPM package
##
SET(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")

SET(CPACK_PACKAGE_GROUP "Development/Libraries")
##
# Source package
##
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}_src_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
SET(CPACK_SOURCE_IGNORE_FILES
    "/build"
    "CMakeFiles"
    "/\\\\.dir"
    "/\\\\.git"
    "/\\\\.gitignore$"
    ".*~$"
    "\\\\.bak$"
    "\\\\.swp$"
    "\\\\.orig$"
    "/\\\\.DS_Store$"
    "/Thumbs\\\\.db"
    "/CMakeLists.txt.user$"
    ${CPACK_SOURCE_IGNORE_FILES})
# Ignore build directories that may be in the source tree
FILE(GLOB_RECURSE CACHES "${CMAKE_SOURCE_DIR}/CMakeCache.txt")

# Call to CPACK
INCLUDE(CPack)
