# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/nsis")

include(Version)
include(CPackIFW)

set(CPACK_GENERATOR "STGZ;TGZ" CACHE STRINGS "STGZ;TGZ;DEB;RPM;productbuild")
set_property(CACHE CPACK_GENERATOR PROPERTY STRINGS STGZ DEB RPM productbuild)
mark_as_advanced(CPACK_GENERATOR)

set(VENDOR_NAME "ArrayFire")
set(LIBRARY_NAME ${PROJECT_NAME})
string(TOLOWER "${LIBRARY_NAME}" APP_LOW_NAME)
set(SITE_URL "www.arrayfire.com")

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
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "A high performance library for parallel computing with an easy-to-use API.")

# Common settings to all packaging tools
set(CPACK_PREFIX_DIR ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_NAME "${LIBRARY_NAME}")
set(CPACK_PACKAGE_VENDOR "${VENDOR_NAME}")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY ${LIBRARY_NAME})
set(CPACK_PACKAGE_CONTACT "ArrayFire Development Group <technical@arrayfire.com>")
set(MY_CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/assets/${APP_LOW_NAME}.ico")

file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/assets/" NATIVE_ASSETS_PATH)
string(REPLACE "\\" "\\\\" NATIVE_ASSETS_PATH  ${NATIVE_ASSETS_PATH})
set(CPACK_AF_ASSETS_DIR "${NATIVE_ASSETS_PATH}")

set(CPACK_PACKAGE_VERSION_MAJOR "${ArrayFire_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ArrayFire_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ArrayFire_VERSION_PATCH}")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${LIBRARY_NAME}")

if(AF_WITH_GRAPHICS)
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${GIT_COMMIT_HASH})
else()
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-no-gl-${GIT_COMMIT_HASH})
endif()

# Platform specific settings for CPACK generators
# - OSX specific
#   - DragNDrop (OSX only)
#   - PackageMaker (OSX only)
#   - OSXX11 (OSX only)
#   - Bundle (OSX only)
# - Windows
#   - NSIS64 Generator
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
elseif(WIN32)
  set(WIN_INSTALL_SOURCE ${PROJECT_SOURCE_DIR}/CMakeModules/nsis)

  set(LICENSE_FILE       "${ArrayFire_SOURCE_DIR}/LICENSE")
  set(LICENSE_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/license.txt")
  configure_file(${LICENSE_FILE} ${LICENSE_FILE_OUT})
  set(CPACK_RESOURCE_FILE_LICENSE ${LICENSE_FILE_OUT})

  #NSIS SPECIFIC VARIABLES
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_DISPLAY_NAME "${LIBRARY_NAME}")
  set(CPACK_NSIS_PACKAGE_NAME "${LIBRARY_NAME}")
  set(CPACK_NSIS_HELP_LINK "${SITE_URL}")
  set(CPACK_NSIS_URL_INFO_ABOUT "${SITE_URL}")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${MY_CPACK_PACKAGE_ICON}")
  if (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  else (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  endif (CMAKE_CL_64)
else()
  set(CPACK_RESOURCE_FILE_LICENSE "${ArrayFire_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${ArrayFire_SOURCE_DIR}/README.md")
endif()

# Set the default components installed in the package
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)

include(CPackComponent)

cpack_add_install_type(Development DISPLAY_NAME "Development")
cpack_add_install_type(Extra DISPLAY_NAME "Extra")
cpack_add_install_type(Runtime DISPLAY_NAME "Runtime")

set(PACKAGE_MKL_DEPS OFF)
set(PACKAGE_GFX_DEPS OFF)

if ((USE_CPU_MKL OR USE_OPENCL_MKL) AND TARGET MKL::MKL)
  set(PACKAGE_MKL_DEPS ON)
  cpack_add_component(mkl_dependencies
    DISPLAY_NAME "Intel MKL Prerequisites"
    DESCRIPTION "Intel MKL libraries required by CPU, OpenCL backends."
    HIDDEN
    INSTALL_TYPES Development Runtime)
endif ()

if (NOT WIN32 AND Forge_FOUND AND NOT AF_USE_SYSTEM_FORGE)
  set(PACKAGE_GFX_DEPS ON)
  cpack_add_component(gfx_dependencies
    DISPLAY_NAME "Graphics prerequisites"
    DESCRIPTION "Graphics library dependencies"
    HIDDEN
    INSTALL_TYPES Development Runtime)
endif ()

cpack_add_component(cuda_dependencies
  DISPLAY_NAME "CUDA Dependencies"
  DESCRIPTION "CUDA Runtime and libraries required for the CUDA backend."
  INSTALL_TYPES Development Runtime)

if (PACKAGE_MKL_DEPS AND PACKAGE_GFX_DEPS)
  cpack_add_component(cpu
    DISPLAY_NAME "CPU Backend"
    DESCRIPTION "This Backend allows you to run ArrayFire code on native CPUs."
    DEPENDS mkl_dependencies gfx_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(opencl
    DISPLAY_NAME "OpenCL Backend"
    DESCRIPTION "This Backend allows you to take advantage of OpenCL capable GPUs to run ArrayFire code. Currently ArrayFire does not support OpenCL for the Intel CPU on OSX."
    DEPENDS mkl_dependencies gfx_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(cuda
    DISPLAY_NAME "CUDA Backend"
    DESCRIPTION "This Backend allows you to take advantage of the CUDA enabled GPUs to run ArrayFire code. Please make sure you have CUDA toolkit installed or install CUDA dependencies component."
    DEPENDS gfx_dependencies cuda_dependencies
    INSTALL_TYPES Development Runtime)
elseif (PACKAGE_MKL_DEPS)
  cpack_add_component(cpu
    DISPLAY_NAME "CPU Backend"
    DESCRIPTION "This Backend allows you to run ArrayFire code on native CPUs."
    DEPENDS mkl_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(opencl
    DISPLAY_NAME "OpenCL Backend"
    DESCRIPTION "This Backend allows you to take advantage of OpenCL capable GPUs to run ArrayFire code. Currently ArrayFire does not support OpenCL for the Intel CPU on OSX."
    DEPENDS mkl_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(cuda
    DISPLAY_NAME "CUDA Backend"
    DESCRIPTION "This Backend allows you to take advantage of the CUDA enabled GPUs to run ArrayFire code. Please make sure you have CUDA toolkit installed or install CUDA dependencies component."
    DEPENDS cuda_dependencies
    INSTALL_TYPES Development Runtime)
elseif (PACKAGE_GFX_DEPS)
  cpack_add_component(cpu
    DISPLAY_NAME "CPU Backend"
    DESCRIPTION "This Backend allows you to run ArrayFire code on native CPUs."
    DEPENDS gfx_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(opencl
    DISPLAY_NAME "OpenCL Backend"
    DESCRIPTION "This Backend allows you to take advantage of OpenCL capable GPUs to run ArrayFire code. Currently ArrayFire does not support OpenCL for the Intel CPU on OSX."
    DEPENDS gfx_dependencies
    INSTALL_TYPES Development Runtime)
  cpack_add_component(cuda
    DISPLAY_NAME "CUDA Backend"
    DESCRIPTION "This Backend allows you to take advantage of the CUDA enabled GPUs to run ArrayFire code. Please make sure you have CUDA toolkit installed or install CUDA dependencies component."
    DEPENDS gfx_dependencies cuda_dependencies
    INSTALL_TYPES Development Runtime)
endif ()

cpack_add_component(unified
  DISPLAY_NAME "Unified Backend"
  DESCRIPTION "This Backend allows you to choose the platform(cpu, cuda, opencl) at runtime. This option requires at least one of the three backends to be installed to work properly."
  INSTALL_TYPES Development Runtime)
cpack_add_component(headers
  DISPLAY_NAME "C/C++ Headers"
  DESCRIPTION "Headers for the ArrayFire Libraries."
  INSTALL_TYPES Development)
cpack_add_component(cmake
  DISPLAY_NAME "CMake Support"
  DESCRIPTION "Configuration files to use ArrayFire using CMake."
  INSTALL_TYPES Development)
cpack_add_component(documentation
  DISPLAY_NAME "Documentation"
  DESCRIPTION "Doxygen documentation"
  INSTALL_TYPES Extra)
cpack_add_component(examples
  DISPLAY_NAME "ArrayFire Examples"
  DESCRIPTION "Various examples using ArrayFire."
  INSTALL_TYPES Extra)
cpack_add_component(licenses
  DISPLAY_NAME "Licenses"
  DESCRIPTION "License files for upstream libraries and ArrayFire."
  REQUIRED)

if (INSTALL_FORGE_DEV)
  cpack_add_component(forge
    DISPLAY_NAME "Forge"
    DESCRIPTION "High Performance Visualization Library"
    INSTALL_TYPES Extra)
endif ()

##
# IFW CPACK generator
# Uses Qt installer framework, cross platform installer generator.
# Uniform installer GUI on all major desktop platforms: Windows, OSX & Linux.
##
set(CPACK_IFW_PACKAGE_TITLE "${CPACK_PACKAGE_NAME}")
set(CPACK_IFW_PACKAGE_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
set(CPACK_IFW_PRODUCT_URL "${SITE_URL}")
set(CPACK_IFW_PACKAGE_ICON "${MY_CPACK_PACKAGE_ICON}")
set(CPACK_IFW_PACKAGE_WINDOW_ICON "${CMAKE_SOURCE_DIR}/assets/${APP_LOW_NAME}_icon.png")
set(CPACK_IFW_PACKAGE_WIZARD_DEFAULT_WIDTH 640)
set(CPACK_IFW_PACKAGE_WIZARD_DEFAULT_HEIGHT 480)
if (WIN32)
    set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "@ApplicationsDirX64@/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
else ()
    set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "/opt/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
endif ()

get_native_path(zlib_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/zlib-libpng License.txt")
get_native_path(boost_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/Boost Software License.txt")
get_native_path(mit_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/MIT License.txt")
get_native_path(fimg_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/FreeImage Public License.txt")
get_native_path(apache_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/Apache-2.0.txt")
get_native_path(sift_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/OpenSIFT License.txt")
get_native_path(bsd3_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/BSD 3-Clause.txt")
get_native_path(issl_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/ISSL License.txt")

if (PACKAGE_MKL_DEPS)
  cpack_ifw_configure_component(mkl_dependencies)
endif ()
if (PACKAGE_GFX_DEPS)
  cpack_ifw_configure_component(gfx_dependencies)
endif ()
cpack_ifw_configure_component(cuda_dependencies)
cpack_ifw_configure_component(cpu)
cpack_ifw_configure_component(cuda)
cpack_ifw_configure_component(opencl)
cpack_ifw_configure_component(unified)
cpack_ifw_configure_component(headers)
cpack_ifw_configure_component(cmake)
cpack_ifw_configure_component(documentation)
cpack_ifw_configure_component(examples)
cpack_ifw_configure_component(licenses FORCED_INSTALLATION
  LICENSES "GLFW" ${zlib_lic_path} "glbinding" ${mit_lic_path} "FreeImage" ${fimg_lic_path}
  "Boost" ${boost_lic_path} "clBLAS, clFFT" ${apache_lic_path} "SIFT" ${sift_lic_path}
  "BSD3" ${bsd3_lic_path} "Intel MKL" ${issl_lic_path}
)
if (INSTALL_FORGE_DEV)
    cpack_ifw_configure_component(forge)
endif ()

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
    ${CPACK_PACKAGE_NAME}_src_${GIT_COMMIT_HASH}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
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

if (WIN32)
  # Configure file with custom definitions for NSIS.
  configure_file(
    ${PROJECT_SOURCE_DIR}/CMakeModules/nsis/NSIS.definitions.nsh.in
    ${CMAKE_CURRENT_BINARY_DIR}/NSIS.definitions.nsh)
endif ()

include(CPack)
