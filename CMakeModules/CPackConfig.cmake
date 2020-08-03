# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# https://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/nsis")

include(Version)
include(CPackIFW)

set(CPACK_GENERATOR "STGZ;TGZ" CACHE STRING "STGZ;TGZ;DEB;RPM;productbuild")
set_property(CACHE CPACK_GENERATOR PROPERTY STRINGS STGZ DEB RPM productbuild)
mark_as_advanced(CPACK_GENERATOR)

set(VENDOR_NAME "ArrayFire")
set(LIBRARY_NAME ${PROJECT_NAME})
string(TOLOWER "${LIBRARY_NAME}" APP_LOW_NAME)
set(SITE_URL "https://arrayfire.com")

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

set(inst_pkg_name ${APP_LOW_NAME})
set(inst_pkg_hash "")
if (WIN32)
  set(inst_pkg_name ${CPACK_PACKAGE_NAME})
  set(inst_pkg_hash "-${GIT_COMMIT_HASH}")
endif ()

set(CPACK_PACKAGE_FILE_NAME "${inst_pkg_name}${inst_pkg_hash}")

# Platform specific settings for CPACK generators
# - OSX specific
#   - DragNDrop (OSX only)
#   - PackageMaker (OSX only)
#   - OSXX11 (OSX only)
#   - Bundle (OSX only)
# - Windows
#   - NSIS64 Generator
if(APPLE)
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/arrayfire")
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

cpack_add_install_type(All DISPLAY_NAME "All Components")
cpack_add_install_type(Development DISPLAY_NAME "Development")
cpack_add_install_type(Extra DISPLAY_NAME "Extra")
cpack_add_install_type(Runtime DISPLAY_NAME "Runtime")

cpack_add_component_group(backends
  DISPLAY_NAME "ArrayFire"
  DESCRIPTION "ArrayFire backend libraries"
  EXPANDED)
cpack_add_component_group(cpu-backend
  DISPLAY_NAME "CPU backend"
  DESCRIPTION "Libraries and dependencies of the CPU backend."
  PARENT_GROUP backends)
cpack_add_component_group(cuda-backend
  DISPLAY_NAME "CUDA backend"
  DESCRIPTION "Libraries and dependencies of the CUDA backend."
  PARENT_GROUP backends)
cpack_add_component_group(opencl-backend
  DISPLAY_NAME "OpenCL backend"
  DESCRIPTION "Libraries and dependencies of the OpenCL backend."
  PARENT_GROUP backends)

set(PACKAGE_MKL_DEPS OFF)

if ((USE_CPU_MKL OR USE_OPENCL_MKL) AND TARGET MKL::Shared)
  set(PACKAGE_MKL_DEPS ON)
  cpack_add_component(mkl_dependencies
    DISPLAY_NAME "Intel MKL"
	DESCRIPTION "Intel Math Kernel Libraries for FFTW, BLAS, and LAPACK routines."
	GROUP backends
    INSTALL_TYPES All Development Runtime)
endif ()

cpack_add_component(common_backend_dependencies
  DISPLAY_NAME "Dependencies"
  DESCRIPTION "Libraries commonly required by all ArrayFire backends."
  GROUP backends
  INSTALL_TYPES All Development Runtime)

cpack_add_component(opencl_dependencies
  DISPLAY_NAME "OpenCL Dependencies"
  DESCRIPTION "Libraries required by the OpenCL backend."
  GROUP opencl-backend
  INSTALL_TYPES All Development Runtime)
if (NOT APPLE) #TODO(pradeep) Remove check after OSX support addition
  cpack_add_component(afopencl_debug_symbols
    DISPLAY_NAME "OpenCL Backend Debug Symbols"
    DESCRIPTION "File containing debug symbols for afopencl dll/so/dylib file"
    GROUP opencl-backend
    DISABLED
    INSTALL_TYPES Development)
endif ()

cpack_add_component(cuda_dependencies
  DISPLAY_NAME "CUDA Dependencies"
  DESCRIPTION "CUDA runtime and libraries required by the CUDA backend."
  GROUP cuda-backend
  INSTALL_TYPES All Development Runtime)
if (NOT APPLE) #TODO(pradeep) Remove check after OSX support addition
  cpack_add_component(afcuda_debug_symbols
    DISPLAY_NAME "CUDA Backend Debug Symbols"
    DESCRIPTION "File containing debug symbols for afcuda dll/so/dylib file"
    GROUP cuda-backend
    DISABLED
    INSTALL_TYPES Development)
endif ()

if (NOT APPLE) #TODO(pradeep) Remove check after OSX support addition
  cpack_add_component(afcpu_debug_symbols
    DISPLAY_NAME "CPU Backend Debug Symbols"
    DESCRIPTION "File containing debug symbols for afcpu dll/so/dylib file"
    GROUP cpu-backend
    DISABLED
    INSTALL_TYPES Development)
endif ()

cpack_add_component(cuda
  DISPLAY_NAME "CUDA Backend"
  DESCRIPTION "The CUDA backend allows you to run ArrayFire code on CUDA-enabled GPUs. Verify that you have the CUDA toolkit installed or install the CUDA dependencies component."
  GROUP cuda-backend
  DEPENDS common_backend_dependencies cuda_dependencies
  INSTALL_TYPES All Development Runtime)

list(APPEND cpu_deps_comps common_backend_dependencies)
list(APPEND ocl_deps_comps common_backend_dependencies)

if (NOT APPLE)
  list(APPEND ocl_deps_comps opencl_dependencies)
endif ()

if (PACKAGE_MKL_DEPS)
  list(APPEND cpu_deps_comps mkl_dependencies)
  list(APPEND ocl_deps_comps mkl_dependencies)
endif ()

cpack_add_component(cpu
  DISPLAY_NAME "CPU Backend"
  DESCRIPTION "The CPU backend allows you to run ArrayFire code on your CPU."
  GROUP cpu-backend
  DEPENDS ${cpu_deps_comps}
  INSTALL_TYPES All Development Runtime)

cpack_add_component(opencl
  DISPLAY_NAME "OpenCL Backend"
  DESCRIPTION "The OpenCL backend allows you to run ArrayFire code on OpenCL-capable GPUs. Note: ArrayFire does not currently support OpenCL for Intel CPUs on OSX."
  GROUP opencl-backend
  DEPENDS ${ocl_deps_comps}
  INSTALL_TYPES All Development Runtime)

if (NOT APPLE) #TODO(pradeep) Remove check after OSX support addition
  cpack_add_component(af_debug_symbols
    DISPLAY_NAME "Unified Backend Debug Symbols"
    DESCRIPTION "File containing debug symbols for af dll/so/dylib file"
    GROUP backends
    DISABLED
    INSTALL_TYPES Development)
endif ()
cpack_add_component(unified
  DISPLAY_NAME "Unified Backend"
  DESCRIPTION "The Unified backend allows you to choose between any of the installed backends (CUDA, OpenCL, or CPU) at runtime."
  GROUP backends
  INSTALL_TYPES All Development Runtime)

cpack_add_component(headers
  DISPLAY_NAME "C/C++ Headers"
  DESCRIPTION "Headers for the ArrayFire libraries."
  GROUP backends
  INSTALL_TYPES All Development)
cpack_add_component(cmake
  DISPLAY_NAME "CMake Support"
  DESCRIPTION "Configuration files to use ArrayFire using CMake."
  INSTALL_TYPES All Development)
cpack_add_component(documentation
  DISPLAY_NAME "Documentation"
  DESCRIPTION "ArrayFire html documentation"
  INSTALL_TYPES All Extra)
cpack_add_component(examples
  DISPLAY_NAME "ArrayFire Examples"
  DESCRIPTION "Various examples using ArrayFire."
  INSTALL_TYPES All Extra)
cpack_add_component(licenses
  DISPLAY_NAME "Licenses"
  DESCRIPTION "License files for ArrayFire and its upstream libraries."
  REQUIRED)

if (AF_INSTALL_FORGE_DEV)
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
get_native_path(fimg_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/FreeImage Public License.txt")
get_native_path(apache_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/Apache-2.0.txt")
get_native_path(sift_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/OpenSIFT License.txt")
get_native_path(bsd3_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/BSD 3-Clause.txt")
get_native_path(issl_lic_path "${CMAKE_SOURCE_DIR}/LICENSES/ISSL License.txt")

cpack_ifw_configure_component_group(backends)
cpack_ifw_configure_component_group(cpu-backend)
cpack_ifw_configure_component_group(cuda-backend)
cpack_ifw_configure_component_group(opencl-backend)
if (PACKAGE_MKL_DEPS)
  cpack_ifw_configure_component(mkl_dependencies)
endif ()
if (NOT APPLE)
  cpack_ifw_configure_component(opencl_dependencies)
endif ()
cpack_ifw_configure_component(common_backend_dependencies)
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
  LICENSES "GLFW" ${zlib_lic_path} "FreeImage" ${fimg_lic_path}
  "Boost" ${boost_lic_path} "clBLAS, clFFT" ${apache_lic_path} "SIFT" ${sift_lic_path}
  "BSD3" ${bsd3_lic_path} "Intel MKL" ${issl_lic_path}
)
if (AF_INSTALL_FORGE_DEV)
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
set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_URL "${SITE_URL}")
if(AF_BUILD_FORGE)
    set(CPACK_RPM_PACKAGE_SUGGESTS "fontconfig-devel, libX11, libXrandr, libXinerama, libXxf86vm, libXcursor, mesa-libGL-devel")
endif()

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
