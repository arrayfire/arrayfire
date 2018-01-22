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
if(AF_WITH_GRAPHICS)
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
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)

include(CPackComponent)

cpack_add_install_type(Development
  DISPLAY_NAME "Development")
cpack_add_install_type(Extra
  DISPLAY_NAME "Extra")
cpack_add_install_type(Runtime
  DISPLAY_NAME "Runtime")

cpack_add_component_group(dependencies
                          DISPLAY_NAME "Dependencies"
                          DESCRIPTION "ArrayFire dependencies"
                          EXPANDED BOLD_TITLE)

cpack_add_component(mkl_dependencies
DISPLAY_NAME "Intel MKL Prerequisites"
DESCRIPTION
"Intel MKL libraries required by CPU/OpenCL backend"
GROUP dependencies
INSTALL_TYPES Development Runtime)

cpack_add_component(cpu_dependencies
DISPLAY_NAME "CPU Dependencies"
DESCRIPTION
"Libraries required for the CPU backend"
GROUP dependencies
DEPENDS mkl_dependencies
INSTALL_TYPES Development Runtime)

cpack_add_component(cuda_dependencies
DISPLAY_NAME "CUDA Dependencies"
DESCRIPTION
"CUDA Runtime and libraries required for the CUDA backend"
GROUP dependencies
INSTALL_TYPES Development Runtime)

cpack_add_component(opencl_dependencies
DISPLAY_NAME "OpenCL Dependencies"
DESCRIPTION
"Libraries required for the OpenCL backend"
GROUP dependencies
DEPENDS mkl_dependencies
INSTALL_TYPES Development Runtime)

cpack_add_component(cpu
DISPLAY_NAME "CPU Backend"
DESCRIPTION
"ArrayFire targeting CPUs."
INSTALL_TYPES Development Runtime)

cpack_add_component(cuda
DISPLAY_NAME "CUDA Backend"
DESCRIPTION
"ArrayFire which targets the CUDA platform. This platform allows you to to take\n"
"advantage of the CUDA enabled GPUs to run ArrayFire code."
INSTALL_TYPES Development Runtime)

cpack_add_component(opencl
DISPLAY_NAME "OpenCL Backend"
DESCRIPTION
"ArrayFire which targets the OpenCL platform. This platform allows you to use\n"
"the ArrayFire library which targets OpenCL devices. CMake config files. NOTE:\n"
"Currently ArrayFire does not support OpenCL for the Intel CPU on OSX."
INSTALL_TYPES Development Runtime)

cpack_add_component(unified
DISPLAY_NAME "Unified Backend"
DESCRIPTION
"This library will allow you to choose the platform(cpu, cuda, opencl) at\n"
"runtime. Also installs the corresponding CMake config files. NOTE: This option\n"
"requires the other platforms to work properly"
INSTALL_TYPES Development Runtime)

cpack_add_component(documentation
DISPLAY_NAME "Documentation"
DESCRIPTION "Doxygen documentation"
INSTALL_TYPES Extra
)

cpack_add_component(headers
DISPLAY_NAME "C/C++ Headers"
DESCRIPTION "Headers for the ArrayFire Libraries."
INSTALL_TYPES Development
)

cpack_add_component(cmake
DISPLAY_NAME "CMake Support"
DESCRIPTION "Configuration files to use ArrayFire using CMake."
INSTALL_TYPES Development
)

cpack_add_component(examples
DISPLAY_NAME "ArrayFire Examples"
DESCRIPTION "Various examples using ArrayFire."
INSTALL_TYPES Extra
)

##
# Debian package
##
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEB_COMPONENT_INSTALL ON)
#set(CMAKE_INSTALL_RPATH /usr/lib;${ArrayFire_BUILD_DIR}/third_party/forge/lib)
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE http://www.arrayfire.com)

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

# Call to CPACK
include(CPack)
