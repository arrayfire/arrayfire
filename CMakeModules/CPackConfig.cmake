# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

include(Version)

set(CPACK_GENERATOR "STGZ;TGZ" CACHE STRINGS "STGZ;TGZ;DEB;RPM;productbuild")
set_property(CACHE CPACK_GENERATOR PROPERTY STRINGS STGZ DEB RPM productbuild)
mark_as_advanced(CPACK_GENERATOR)

# Common settings to all packaging tools
set(CPACK_PREFIX_DIR ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_NAME "arrayfire")
set(CPACK_PACKAGE_VENDOR "ArrayFire")
set(CPACK_PACKAGE_CONTACT "ArrayFire Development Group <technical@arrayfire.com>")

set(CPACK_PACKAGE_VERSION ${ArrayFire_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR "${ArrayFire_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ArrayFire_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ArrayFire_VERSION_PATCH}")
if(BUILD_GRAPHICS)
    set(CPACK_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})
else()
    set(CPACK_PACKAGE_FILE_NAME
        ${CPACK_PACKAGE_NAME}-no-gl-${CPACK_PACKAGE_VERSION})
endif()

if(APPLE)
  set(OSX_INSTALL_SOURCE ${PROJECT_SOURCE_DIR}/CMakeModules/osx_install)
  set(WELCOME_FILE       "${OSX_INSTALL_SOURCE}/welcome.html.in")
  set(WELCOME_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/welcome.html")

  set(README_FILE       "${OSX_INSTALL_SOURCE}/readme.html.in")
  set(README_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/readme.html")

  set(LICENSE_FILE       "${ArrayFire_SOURCE_DIR}/LICENSE")
  set(LICENSE_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/license.txt")

  set(AF_TITLE    "ArrayFire ${AF_VERSION}")
  configure_file(${WELCOME_FILE} ${WELCOME_FILE_OUT})
  configure_file(${README_FILE} ${README_FILE_OUT})
  configure_file(${LICENSE_FILE} ${LICENSE_FILE_OUT})
  set(CPACK_RESOURCE_FILE_LICENSE ${LICENSE_FILE_OUT})
  set(CPACK_RESOURCE_FILE_README ${README_FILE_OUT})
  set(CPACK_RESOURCE_FILE_WELCOME ${WELCOME_FILE_OUT})
else()
  set(CPACK_RESOURCE_FILE_LICENSE "${ArrayFire_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${ArrayFire_SOURCE_DIR}/README.md")
endif()

# Long description of the package
set(CPACK_PACKAGE_DESCRIPTION
"ArrayFire is a high performance software library for parallel computing
with an easy-to-use API. Its array based function set makes parallel
programming simple.

ArrayFire's multiple backends (CUDA, OpenCL and native CPU) make it
platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel
computing code, saving you valuable time and lowering development costs.")

# Short description of the package
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A high performance library for parallel computing with an easy-to-use API.")

# Set the default components installed in the package
set(CPACK_COMPONENTS_ALL cpu cuda opencl unified headers documentation cmake examples)

include(CPackComponent)
cpack_add_component_group(libraries
DISPLAY_NAME "Libraries"
DESCRIPTION "ArrayFire libraries"
EXPANDED BOLD_TITLE)

cpack_add_component(cpu
DISPLAY_NAME "CPU Backend"
DESCRIPTION
"ArrayFire targeting CPUs. Also installs the corresponding CMake config files."
GROUP libraries)

cpack_add_component(cuda
DISPLAY_NAME "CUDA Backend"
DESCRIPTION
"ArrayFire which targets the CUDA platform. This platform allows you to to take "
"advantage of the CUDA enabled GPUs to run ArrayFire code. Also installs the "
"corresponding CMake config files."
GROUP libraries)

cpack_add_component(opencl
DISPLAY_NAME "OpenCL Backend"
DESCRIPTION
"ArrayFire which targets the OpenCL platform. This platform allows you to use the "
"ArrayFire library which targets OpenCL devices. Also installs the corresponding "
"CMake config files. NOTE: Currently ArrayFire does not support OpenCL for the "
"Intel CPU on OSX."
GROUP libraries)

cpack_add_component(unified
DISPLAY_NAME "Unified Backend"
DESCRIPTION
"This library will allow you to choose the platform(cpu, cuda, opencl) at "
"runtime. Also installs the corresponding CMake config files. NOTE: This option "
"requires the other platforms to work properly"
#DEPENDS "cpu;cuda;opencl"
GROUP libraries)

cpack_add_component(documentation
DISPLAY_NAME "Documentation"
DESCRIPTION "Doxygen documentation"
)

cpack_add_component(headers
DISPLAY_NAME "C/C++ Headers"
DESCRIPTION "Headers for the ArrayFire Libraries."
)

cpack_add_component(cmake
DISPLAY_NAME "CMake Support"
DESCRIPTION "Configuration files to use ArrayFire using CMake."
)

cpack_add_component(examples
DISPLAY_NAME "ArrayFire Examples"
DESCRIPTION "Various examples using ArrayFire."
)

##
# Debian package
##
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${PROCESSOR_ARCHITECTURE})

##
# RPM package
##
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")

set(CPACK_PACKAGE_GROUP "Development/Libraries")
##
# Source package
##
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}_src_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
set(CPACK_SOURCE_IGNORE_FILES
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
file(GLOB_RECURSE CACHES "${CMAKE_SOURCE_DIR}/CMakeCache.txt")

SET(CPACK_WIX_LICENSE_RTF "${PROJECT_SOURCE_DIR}/LICENSE.rtf")
SET(CPACK_WIX_PATCH_FILE "${PROJECT_SOURCE_DIR}/wix/WIXPatch.wxs")
SET(CPACK_WIX_UPGRADE_GUID "FF9E2D77-CDC7-4D24-8B7B-99D66EDEE862")

# Call to CPACK
include(CPack)
