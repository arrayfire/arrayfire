##
# RPM package
##

include(CMakeParseArguments)

#TODO(umar): remove, should be set form command line
# set(CPACK_RPM_PACKAGE_RELEASE 1)

set(CPACK_RPM_FILE_NAME "%{name}-%{version}-%{release}%{?dist}.%{_arch}.rpm")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_URL "${SITE_URL}")
#set(CPACK_RPM_CHANGELOG_FILE "${ArrayFire_SOURCE_DIR}/docs/pages/release_notes.md")

set(CPACK_COMPONENTS_ALL
  arrayfire
  cpu
  cpu-dev
  cpu-cmake
  cuda
  cuda-dev
  cuda-cmake
  opencl
  opencl-dev
  opencl-cmake
  unified
  unified-dev
  unified-cmake
  headers
  cmake
  examples
  documentation
  )

macro(af_rpm_component)
  cmake_parse_arguments(RC "" "COMPONENT;NAME;SUMMARY;DESCRIPTION" "REQUIRES;OPTIONAL" ${ARGN})
  cpack_add_component(${RC_COMPONENT}
    #DISPLAY_NAME
    DESCRIPTION ${RC_DESCRIPTION})

  string(REPLACE ";" ", " REQ "${RC_REQUIRES}")
  string(REPLACE ";" ", " OPT "${RC_OPTIONAL}")
  string(TOUPPER ${RC_COMPONENT} COMPONENT_UPPER)

  if(RC_NAME)
      set(CPACK_RPM_${COMPONENT_UPPER}_PACKAGE_NAME "${RC_NAME}")
  endif()
  # NOTE: Does not work in CentOS 6
  #set(CPACK_RPM_${COMPONENT_UPPER}_PACKAGE_SUGGESTS ${OPT})
  set(CPACK_RPM_${COMPONENT_UPPER}_SUMMARY ${SUMMARY})
  set(CPACK_RPM_${COMPONENT_UPPER}_PACKAGE_REQUIRES "${REQ}")
endmacro()

set(CPACK_RPM_MAIN_COMPONENT arrayfire)
af_rpm_component(
  COMPONENT arrayfire
  REQUIRES arrayfire-cpu-dev arrayfire-cuda-dev arrayfire-opencl-dev arrayfire-examples arrayfire-doc
  SUMMARY  "ArrayFire high performance library"
  DESCRIPTION  "ArrayFire
ArrayFire is a general-purpose library that simplifies software
development that targets parallel and massively-parallel architectures
including CPUs, GPUs, and other hardware acceleration devices.")

af_rpm_component(
  COMPONENT cpu
  NAME arrayfire-cpu-runtime
  OPTIONAL forge-runtime
  SUMMARY "ArrayFire CPU backend shared libraries"
  DESCRIPTION "ArrayFire CPU backend shared libraries")

af_rpm_component(
  COMPONENT cpu-dev
  REQUIRES arrayfire-cpu-runtime arrayfire-headers arrayfire-cmake arrayfire-cpu-cmake
  SUMMARY  "ArrayFire CPU Backend development files"
  DESCRIPTION  "ArrayFire CPU Backend development files")

af_rpm_component(
  COMPONENT cuda
  NAME arrayfire-cuda-runtime
  OPTIONAL forge-runtime
  SUMMARY "ArrayFire CUDA backend shared libraries"
  DESCRIPTION "ArrayFire CUDA backend shared libraries")

af_rpm_component(
  COMPONENT cuda-dev
  REQUIRES arrayfire-cuda-runtime arrayfire-headers arrayfire-cmake arrayfire-cuda-cmake
  SUMMARY  "ArrayFire CUDA Backend development files"
  DESCRIPTION  "ArrayFire CUDA Backend development files")

af_rpm_component(
  COMPONENT opencl
  NAME arrayfire-opencl-runtime
  OPTIONAL forge-runtime
  SUMMARY "ArrayFire OpenCL backend shared libraries"
  DESCRIPTION "ArrayFire OpenCL backend shared libraries")

af_rpm_component(
  COMPONENT opencl-dev
  REQUIRES arrayfire-opencl-runtime arrayfire-headers arrayfire-cmake arrayfire-opencl-cmake
  SUMMARY  "ArrayFire OpenCL Backend development files"
  DESCRIPTION  "ArrayFire OpenCL Backend development files")

af_rpm_component(
  COMPONENT unified
  NAME arrayfire-unified-runtime
  OPTIONAL forge-runtime
  SUMMARY "ArrayFire Unified backend shared libraries."
  DESCRIPTION "ArrayFire Unified backend shared libraries. Requires other backends to function.")

af_rpm_component(
  COMPONENT unified-dev
  REQUIRES arrayfire-unified-runtime arrayfire-headers arrayfire-cmake arrayfire-unified-cmake
  OPTIONAL forge-runtime
  SUMMARY  "ArrayFire Unified Backend development files"
  DESCRIPTION  "ArrayFire Unified Backend development files")

af_rpm_component(
  COMPONENT documentation
  NAME arrayfire-doc
  SUMMARY  "ArrayFire Documentation"
  DESCRIPTION  "ArrayFire Doxygen Documentation")

# NOTE: These commands do not work well for forge. The
# version or the package name is incorrect when you perform
# the package creation this way. The CPACK_PACKAGE_VERSION
# does not extend to components. The forge rpm package will
# need to be created in the forge project.

#af_rpm_component(
#  COMPONENT forge-lib
#  NAME forge-runtime
#  DESCRIPTION  )

#cpack_add_component(forge-lib
#    DESCRIPTION "Forge runtime libraries")
#
#set(CPACK_RPM_FORGE-LIB_PACKAGE_NAME "forge-runtime")
#get_target_property(Forge_VERSION forge VERSION)
##set(CPACK_RPM_FORGE-RUNTIME_FILE_NAME "forge-runtime-${Forge_VERSION}-%{release}%{?dist}.%{_arch}.rpm")
#set(CPACK_RPM_FORGE-LIB_FILE_NAME "forge-runtime-${Forge_VERSION}-%{release}%{?dist}.%{_arch}.rpm")

# Does not work in CentOS 6
#set(CPACK_RPM_${COMPONENT_UPPER}_PACKAGE_SUGGESTS ${OPT})
#set(CPACK_RPM_FORGE-LIB_SUMMARY "Forge runtime librariers")
#set(CPACK_RPM_FORGE-LIB_VERSION ${Forge_VERSION})
#set(CPACK_RPM_FORGE-LIB_PACKAGE_REQUIRES "${REQ}")
