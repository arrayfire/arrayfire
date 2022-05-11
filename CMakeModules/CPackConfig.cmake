# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# https://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/nsis")

include(Version)

set(CPACK_THREADS 8)

set(CPACK_GENERATOR "STGZ;TGZ" CACHE STRING "STGZ;TGZ;DEB;RPM;productbuild")
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
set(CPACK_PACKAGE_CONTACT "ArrayFire <technical@arrayfire.com>")
set(MY_CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/assets/${APP_LOW_NAME}.ico")

file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/assets/" NATIVE_ASSETS_PATH)
string(REPLACE "\\" "\\\\" NATIVE_ASSETS_PATH  ${NATIVE_ASSETS_PATH})
set(CPACK_AF_ASSETS_DIR "${NATIVE_ASSETS_PATH}")

set(CPACK_PACKAGE_VERSION_MAJOR "${ArrayFire_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ArrayFire_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ArrayFire_VERSION_PATCH}")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${LIBRARY_NAME}")

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_DEBUGINFO_PACKAGE OFF)
set(CPACK_DEBIAN_PACKAGE_DEBUG ON)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY ">=")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE http://www.arrayfire.com)
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
set(CPACK_DEBIAN_COMPRESSION_TYPE xz)
set(CPACK_DEBIAN_DEBUGINFO_PACKAGE ON)

# Creates a variable from a ArrayFire variable so that it can be passed
# into cpack project file. This is done by prepending CPACK_ before the
# variable name
macro(to_cpack_variable variable)
  set(CPACK_${variable} ${${variable}})
endmacro()

to_cpack_variable(AF_COMPUTE_LIBRARY)
to_cpack_variable(ArrayFire_SOURCE_DIR)
to_cpack_variable(ArrayFire_BINARY_DIR)
to_cpack_variable(CUDA_VERSION_MAJOR)
to_cpack_variable(CUDA_VERSION_MINOR)

# Create a arrayfire component so that Debian package has a top level
# package that installs all the backends. This package needs to have
# some files associated with it so that it doesn't get deleted by
# APT after its installed.
file(WRITE ${ArrayFire_BINARY_DIR}/arrayfire_version.txt ${ArrayFire_VERSION})
install(FILES ${ArrayFire_BINARY_DIR}/arrayfire_version.txt
	DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}
  COMPONENT arrayfire)

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
  set(CPACK_NSIS_COMPRESSOR "lzma")
  if (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  else (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  endif (CMAKE_CL_64)
else()
  set(CPACK_RESOURCE_FILE_LICENSE "${ArrayFire_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${ArrayFire_SOURCE_DIR}/README.md")
endif()

set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_SOURCE_DIR}/CMakeModules/CPackProjectConfig.cmake")

include(CPack)
