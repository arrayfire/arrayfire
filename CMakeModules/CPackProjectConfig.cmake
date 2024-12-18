
include(CPackIFW)
include(CPackComponent)

# Only install the components created using the af_component macro
set(CPACK_COMPONENTS_ALL "")

# This is necessary if you don't have a cuda driver installed on your system
# but you are still building the cuda package. You need the libcuda.so library
# which is installed by the driver. This tell the dpkg-shlibs to ignore
# this library because it is a private library
set (CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS
  "/usr/local/cuda-${CPACK_CUDA_VERSION_MAJOR}.${CPACK_CUDA_VERSION_MINOR}/lib64/stubs")


# Create an ArrayFire component with a set of properties for each package manager
# This function sets all the variables for each component in ArrayFire.
#
# ``COMPONENT``
# The name of the ArrayFire component used in the install(XXX) commands
#
# ``DISPLAY_NAME``
# The name that will appear in the GUI installers for this component
#
# ``SUMMARY``
# A short one line summary of the package
#
# ``DESCRIPTION``
# A longer description of the package
#
# ``GROUP``
# Used to combine packages in GUI installers. Ignored in DEB and RPM installers
#
# ``DEB_PACKAGE_NAME``
# Name of the package for the DEB installers. This is the first component of the
# file name.
#
# ``DEB_PROVIDES``
# The virtual packages provided by the deb package. This is a higher level name
# of the file that can be used across version numbers. also includes the version
# information about the package
#
# ``DEB_REPLACES``
# The packages and virtual packages this will replace. Used if there is a package
# that is installed as part of the base debian installation
#
# ``REQUIRES``
# The components required for the GUI installers
#
# ``OPTIONAL``
# Optional packages that this component can use.
#
# ``INSTALL_TYPE``
# A group of components that will be selected in GUI installers from a drop down
#
# ``DEB_REQUIRES``
# Set of packages required by the debian package. This is slighly different from
# REQUIRES because it also takes into account external dependencies that can be
# installed by apt
#
# ``DEB_OPTIONAL``
# Same as OPTIONAL but for debian packages
#
# ``DEB_RECOMMENDS``
# Packages that should be installed but are not required. These packages will
# be installed by default but if removed will not also delete this package
#
# ``HIDDEN``
# If set, the package will not appear in the GUI installers like NSIS. Usually
# components that install dependencies
macro(af_component)
  cmake_parse_arguments(RC
    "HIDDEN;DISABLED;DEB_USE_SHLIBDEPS;DEB_ADD_POSTINST"
    "COMPONENT;DISPLAY_NAME;SUMMARY;DESCRIPTION;GROUP;DEB_PACKAGE_NAME;DEB_PROVIDES;DEB_REPLACES"
    "REQUIRES;OPTIONAL;INSTALL_TYPES;DEB_REQUIRES;DEB_OPTIONAL;DEB_RECOMMENDS" ${ARGN})

  list(APPEND CPACK_COMPONENTS_ALL ${RC_COMPONENT})

  string(TOUPPER ${RC_COMPONENT} COMPONENT_UPPER)
  string(REPLACE ";" ", " DEB_REQ "${RC_DEB_REQUIRES}")
  string(REPLACE ";" ", " DEB_REC "${RC_DEB_RECOMMENDS}")
  string(REPLACE ";" ", " DEB_OPT "${RC_DEB_OPTIONAL}")
  string(REPLACE ";" ", " DEB_PROVIDES "${RC_DEB_PROVIDES}")

  if(CPACK_GENERATOR MATCHES "DEB")
    cpack_add_component(${RC_COMPONENT}
      DISPLAY_NAME "${RC_DISPLAY_NAME}"
      INSTALL_TYPES ${RC_INSTALL_TYPES}
      DESCRIPTION ${RC_DESCRIPTION})

    if(RC_DEB_RECOMMENDS)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_RECOMMENDS ${DEB_REC})
    endif()

    if(RC_DEB_PACKAGE_NAME)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_NAME "${RC_DEB_PACKAGE_NAME}")
    endif()

    set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_SUGGESTS ${DEB_OPT})

    if(RC_DEB_REQUIRES)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_DEPENDS "${DEB_REQ}")
    endif()

    if(RC_DEB_USE_SHLIBDEPS)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_SHLIBDEPS ON)
    else()
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_SHLIBDEPS OFF)
    endif()

    if(RC_DEB_PROVIDES)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_PROVIDES ${DEB_PROVIDES})
    endif()

    if(RC_DEB_REPLACES)
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_REPLACES ${RC_DEB_REPLACES})
      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_CONFLICTS ${RC_DEB_REPLACES})
    endif()

    if(RC_DEB_ADD_POSTINST)
      configure_file(
        "${CPACK_ArrayFire_SOURCE_DIR}/CMakeModules/debian/postinst"
        "${CPACK_ArrayFire_BINARY_DIR}/cpack/${COMPONENT_UPPER}/postinst")

      set(CPACK_DEBIAN_${COMPONENT_UPPER}_PACKAGE_CONTROL_EXTRA
        "${CPACK_ArrayFire_BINARY_DIR}/cpack/${COMPONENT_UPPER}/postinst")
    endif()
  else()
    cpack_add_component(${RC_COMPONENT}
      DISPLAY_NAME "${RC_DISPLAY_NAME}"
      DEPENDS ${RC_REQUIRES}
      GROUP ${RC_GROUP}
      INSTALL_TYPES ${RC_INSTALL_TYPES}
      DESCRIPTION ${RC_DESCRIPTION})
  endif()

  set(CPACK_COMPONENT_${RC_COMPONENT}_DESCRIPTION_SUMMARY ${RC_SUMMARY})
  set(CPACK_COMPONENT_${COMPONENT_UPPER}_DESCRIPTION ${RC_DESCRIPTION})

  set(CPACK_COMPONENT_${COMPONENT_UPPER}_HIDDEN ${RC_HIDDEN})
  set(CPACK_COMPONENT_${COMPONENT_UPPER}_DISABLED ${RC_DISABLED})

  # Does not work with RPM for some reason using
  # CPACK_RPM_${COMPONENT_UPPER}_PACKAGE_REQUIRES  instead

endmacro()

cpack_add_install_type(All DISPLAY_NAME "All Components")
cpack_add_install_type(Development DISPLAY_NAME "Development")
cpack_add_install_type(Runtime DISPLAY_NAME "Runtime")

# Groups on debian packages will combine all the packages into one
# debian component
if(NOT CPACK_GENERATOR MATCHES "DEB")
  cpack_add_component_group(afruntime
    DISPLAY_NAME "ArrayFire Runtime"
    DESCRIPTION "ArrayFire runtime libraries")

  cpack_add_component_group(afdevelopment
    DISPLAY_NAME "ArrayFire Development"
    DESCRIPTION "ArrayFire development files including headers and configuration files"
    EXPANDED)

  cpack_add_component_group(debug
    DISPLAY_NAME "ArrayFire Debug Symbols"
    DESCRIPTION "ArrayFire Debug symbols")
endif()

set(arrayfire_cuda_runtime_name "CUDA Runtime(${CPACK_CUDA_VERSION_MAJOR}.${CPACK_CUDA_VERSION_MINOR})")
set(arrayfire_cuda_dev_name "CUDA Dev")

if(CPACK_GENERATOR MATCHES "DEB")
  af_component(
    COMPONENT arrayfire
    REQUIRES cpu_dev cuda_dev opencl_dev examples documentation
    SUMMARY  "ArrayFire high performance library"
    DESCRIPTION  "ArrayFire
ArrayFire is a general-purpose library that simplifies software
development that targets parallel and massively-parallel architectures
including CPUs, GPUs, and other hardware acceleration devices."

    DEB_PACKAGE_NAME arrayfire
    DEB_REQUIRES arrayfire-cpu3-dev
                 arrayfire-headers

    DEB_RECOMMENDS arrayfire-cuda3-dev
                   arrayfire-opencl3-dev
                   arrayfire-unified3-dev
                   arrayfire-examples
                   arrayfire-cmake
                   arrayfire-doc
  )
endif()


list(APPEND cpu_deps_comps common_backend_dependencies)
list(APPEND ocl_deps_comps common_backend_dependencies)

if (NOT APPLE)
  list(APPEND ocl_deps_comps opencl_dependencies)
endif ()

set(PACKAGE_MKL_DEPS OFF)

if(CPACK_CUDA_VERSION_MAJOR STREQUAL "10" AND CPACK_GENERATOR MATCHES "DEB")
  set(deb_cuda_runtime_requirements "libcublas${CPACK_CUDA_VERSION_MAJOR}")
elseif(CPACK_CUDA_VERSION_MAJOR STREQUAL "11" AND CPACK_GENERATOR MATCHES "DEB")
  set(deb_cuda_runtime_requirements "libcublas-${CPACK_CUDA_VERSION_MAJOR}-${CPACK_CUDA_VERSION_MINOR}")
elseif(CPACK_GENERATOR MATCHES "DEB")
  message(FATAL_ERROR "THIS CUDA VERSION NOT ADDRESSED FOR DEBIN PACKAGES")
endif()

if (CPACK_AF_COMPUTE_LIBRARY STREQUAL "Intel-MKL")
  set(PACKAGE_MKL_DEPS ON)
  if(NOT CPACK_GENERATOR STREQUAL "DEB")
    af_component(
      COMPONENT mkl_dependencies
      DISPLAY_NAME "Intel MKL Libraries"
            DESCRIPTION "Intel Math Kernel Libraries for FFTW, BLAS, and LAPACK routines."
      HIDDEN
      INSTALL_TYPES All Runtime)
    list(APPEND cpu_deps_comps mkl_dependencies)
    list(APPEND ocl_deps_comps mkl_dependencies)
  endif()
  set(deb_opencl_runtime_package_name arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR}-mkl)
  set(deb_opencl_runtime_requirements "intel-mkl-core-rt-2020.0-166, intel-mkl-gnu-rt-2020.0-166")
  set(deb_cpu_runtime_package_name arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-mkl)
  set(deb_cpu_runtime_requirements "intel-mkl-core-rt-2020.0-166, intel-mkl-gnu-rt-2020.0-166")
else()
  # OpenCL and CPU runtime dependencies are detected using
  # SHLIBDEPS
  set(deb_opencl_runtime_package_name arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR}-openblas)
  set(deb_opencl_runtime_requirements "")
  set(deb_cpu_runtime_package_name arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-openblas)
  set(deb_cpu_runtime_requirements "")
endif ()

af_component(
  COMPONENT cpu
  DISPLAY_NAME "CPU Runtime"
  SUMMARY "ArrayFire CPU backend shared libraries"
  DESCRIPTION "ArrayFire CPU backend shared libraries"
  OPTIONAL forge
  GROUP afruntime
  REQUIRES ${cpu_deps_comps} licenses
  INSTALL_TYPES All Runtime

  DEB_PACKAGE_NAME ${deb_cpu_runtime_package_name}
  DEB_REQUIRES ${deb_cpu_runtime_requirements}
  DEB_PROVIDES "arrayfire-cpu (= ${CPACK_PACKAGE_VERSION}), arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION}), libarrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-cpu, arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION})"
  DEB_USE_SHLIBDEPS
  DEB_ADD_POSTINST
  DEB_OPTIONAL forge libfreeimage3
)

af_component(
  COMPONENT cpu_dev
  DISPLAY_NAME "CPU Dev"
  SUMMARY  "ArrayFire CPU backend development files"
  DESCRIPTION  "ArrayFire CPU backend development files"
  REQUIRES cpu headers cmake
  GROUP afdevelopment
  INSTALL_TYPES All Development

  DEB_PACKAGE_NAME arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-dev
  DEB_PROVIDES "arrayfire-cpu-dev (= ${CPACK_PACKAGE_VERSION}), arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-dev (= ${CPACK_PACKAGE_VERSION}), libarrayfire-cpu-dev (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-cpu-dev (<< ${CPACK_PACKAGE_VERSION}), arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-dev (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-cpu3-dev (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-openblas (>= ${CPACK_PACKAGE_VERSION}) | arrayfire-cpu${CPACK_PACKAGE_VERSION_MAJOR}-mkl (>= ${CPACK_PACKAGE_VERSION}), arrayfire-headers (>= ${CPACK_PACKAGE_VERSION})"
  DEB_RECOMMENDS "arrayfire-cmake (>= ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL "cmake (>= 3.0)"
)

af_component(
  COMPONENT cuda
  DISPLAY_NAME "${arrayfire_cuda_runtime_name}"
  SUMMARY "ArrayFire CUDA backend shared libraries"
  DESCRIPTION "ArrayFire CUDA backend shared libraries"
  OPTIONAL forge
  REQUIRES common_backend_dependencies cuda_dependencies licenses
  GROUP afruntime
  INSTALL_TYPES All Runtime

  DEB_PACKAGE_NAME arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR}-cuda-${CPACK_CUDA_VERSION_MAJOR}-${CPACK_CUDA_VERSION_MINOR}
  DEB_REQUIRES ${deb_cuda_runtime_requirements}
  DEB_ADD_POSTINST
  DEB_USE_SHLIBDEPS
  DEB_PROVIDES "arrayfire-cuda (= ${CPACK_PACKAGE_VERSION}), arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION}), libarrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-cuda (<< ${CPACK_PACKAGE_VERSION}), arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL libcudnn8 forge libfreeimage3
)

af_component(
  COMPONENT cuda_dev
  DISPLAY_NAME "${arrayfire_cuda_dev_name}"
  SUMMARY  "ArrayFire CUDA backend development files"
  DESCRIPTION  "ArrayFire CUDA backend development files"
  REQUIRES cuda headers cmake
  GROUP afdevelopment
  INSTALL_TYPES All Development

  DEB_PACKAGE_NAME arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR}-dev
  DEB_PROVIDES "arrayfire-cuda-dev (= ${CPACK_PACKAGE_VERSION}), arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR}-dev (= ${CPACK_PACKAGE_VERSION}), libarrayfire-cuda-dev (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-cuda-dev (<< ${CPACK_PACKAGE_VERSION}), arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR}-dev (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-cuda${CPACK_PACKAGE_VERSION_MAJOR} (>= ${CPACK_PACKAGE_VERSION}), arrayfire-headers (>= ${CPACK_PACKAGE_VERSION})"
  DEB_RECOMMENDS "arrayfire-cmake (>= ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL "cmake (>= 3.0)"
)

af_component(
  COMPONENT opencl
  DISPLAY_NAME "OpenCL Runtime"
  SUMMARY "ArrayFire OpenCL backend shared libraries"
  DESCRIPTION "ArrayFire OpenCL backend shared libraries"
  REQUIRES ${opencl_deps_comps} licenses
  OPTIONAL forge
  GROUP afruntime
  INSTALL_TYPES All Runtime

  DEB_PACKAGE_NAME ${deb_opencl_runtime_package_name}
  DEB_PROVIDES "arrayfire-opencl (= ${CPACK_PACKAGE_VERSION}), arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION}), libarrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-opencl (<< ${CPACK_PACKAGE_VERSION}), arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES ${deb_opencl_runtime_requirements}
  DEB_USE_SHLIBDEPS
  DEB_ADD_POSTINST
  DEB_OPTIONAL forge libfreeimage3
)

af_component(
  COMPONENT opencl_dev
  DISPLAY_NAME "OpenCL Dev"
  SUMMARY  "ArrayFire OpenCL backend development files"
  DESCRIPTION  "ArrayFire OpenCL backend development files"
  REQUIRES opencl headers cmake
  GROUP afdevelopment
  INSTALL_TYPES All Development

  DEB_PACKAGE_NAME arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR}-dev
  DEB_PROVIDES "arrayfire-opencl-dev (= ${CPACK_PACKAGE_VERSION}), arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR}-dev (= ${CPACK_PACKAGE_VERSION}), libarrayfire-opencl-dev (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-opencl-dev (<< ${CPACK_PACKAGE_VERSION}), arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR}-dev (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-opencl-dev (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-opencl${CPACK_PACKAGE_VERSION_MAJOR} (>= ${CPACK_PACKAGE_VERSION}), arrayfire-headers (>= ${CPACK_PACKAGE_VERSION})"
  DEB_RECOMMENDS "arrayfire-cmake (>= ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL "cmake (>= 3.0)"
)

af_component(
  COMPONENT oneapi
  DISPLAY_NAME "oneAPI Runtime"
  SUMMARY "ArrayFire oneAPI backend shared libraries"
  DESCRIPTION "ArrayFire oneAPI backend shared libraries"
  REQUIRES ${oneapi_deps_comps} licenses
  OPTIONAL forge
  GROUP afruntime
  INSTALL_TYPES All Runtime

  DEB_PACKAGE_NAME ${deb_oneapi_runtime_package_name}
  DEB_PROVIDES "arrayfire-oneapi (= ${CPACK_PACKAGE_VERSION}), arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION}), libarrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-oneapi (<< ${CPACK_PACKAGE_VERSION}), arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES ${deb_oneapi_runtime_requirements}
  DEB_USE_SHLIBDEPS
  DEB_ADD_POSTINST
  DEB_OPTIONAL forge libfreeimage3
)

af_component(
  COMPONENT oneapi_dev
  DISPLAY_NAME "oneAPI Dev"
  SUMMARY  "ArrayFire oneAPI backend development files"
  DESCRIPTION  "ArrayFire oneAPI backend development files"
  REQUIRES oneapi headers cmake
  GROUP afdevelopment
  INSTALL_TYPES All Development

  DEB_PACKAGE_NAME arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR}-dev
  DEB_PROVIDES "arrayfire-oneapi-dev (= ${CPACK_PACKAGE_VERSION}), arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR}-dev (= ${CPACK_PACKAGE_VERSION}), libarrayfire-oneapi-dev (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-oneapi-dev (<< ${CPACK_PACKAGE_VERSION}), arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR}-dev (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-oneapi-dev (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-oneapi${CPACK_PACKAGE_VERSION_MAJOR} (>= ${CPACK_PACKAGE_VERSION}), arrayfire-headers (>= ${CPACK_PACKAGE_VERSION})"
  DEB_RECOMMENDS "arrayfire-cmake (>= ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL "cmake (>= 3.0)"
)

af_component(
  COMPONENT unified
  DISPLAY_NAME "Unified Runtime"
  SUMMARY "ArrayFire Unified backend shared libraries."
  DESCRIPTION "ArrayFire Unified backend shared libraries. Requires other backends to function."
  OPTIONAL forge
  REQUIRES licenses
  GROUP afruntime
  INSTALL_TYPES All Runtime

  DEB_PACKAGE_NAME arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR}
  DEB_PROVIDES "arrayfire-unified (= ${CPACK_PACKAGE_VERSION}), arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION}), libarrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR} (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-unified (<< ${CPACK_PACKAGE_VERSION}), arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR} (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-cpu (>= ${CPACK_PACKAGE_VERSION}) | arrayfire-cuda (>= ${CPACK_PACKAGE_VERSION}) | arrayfire-opencl (>= ${CPACK_PACKAGE_VERSION})"
  DEB_USE_SHLIBDEPS
)

af_component(
  COMPONENT unified_dev
  DISPLAY_NAME "Unified Dev"
  SUMMARY  "ArrayFire Unified backend development files"
  DESCRIPTION  "ArrayFire Unified backend development files"
  REQUIRES unified headers cmake
  OPTIONAL forge
  GROUP afdevelopment
  INSTALL_TYPES All Development

  DEB_PACKAGE_NAME arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR}-dev
  DEB_PROVIDES "arrayfire-unified-dev (= ${CPACK_PACKAGE_VERSION}), arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR}-dev (= ${CPACK_PACKAGE_VERSION}), libarrayfire-unified-dev (= ${CPACK_PACKAGE_VERSION})"
  DEB_REPLACES "arrayfire-unified-dev (<< ${CPACK_PACKAGE_VERSION}), arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR}-dev (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-unified-dev (<< ${CPACK_PACKAGE_VERSION})"
  DEB_REQUIRES "arrayfire-unified${CPACK_PACKAGE_VERSION_MAJOR} (>= ${CPACK_PACKAGE_VERSION})"
  DEB_RECOMMENDS "arrayfire-cmake (>= ${CPACK_PACKAGE_VERSION})"
  DEB_OPTIONAL "cmake (>= 3.0)"
)

af_component(
  COMPONENT documentation
  DISPLAY_NAME "Documentation"
  SUMMARY  "ArrayFire Documentation"
  INSTALL_TYPES All
  DESCRIPTION  "ArrayFire Doxygen Documentation"

  DEB_PACKAGE_NAME arrayfire-doc
  DEB_REPLACES "arrayfire-doc (<< ${CPACK_PACKAGE_VERSION}), libarrayfire-doc (<< ${CPACK_PACKAGE_VERSION})"
)

af_component(
  COMPONENT headers
  DISPLAY_NAME "C/C++ Headers"
  HIDDEN
  INSTALL_TYPES All Development
  DESCRIPTION "Headers for the ArrayFire libraries.")

af_component(
  COMPONENT examples
  DISPLAY_NAME "ArrayFire Examples"
  INSTALL_TYPES All
  DESCRIPTION "Various examples using ArrayFire.")

af_component(
  COMPONENT cmake
  DISPLAY_NAME "CMake Files"
  HIDDEN
  INSTALL_TYPES All Development
  DESCRIPTION "Configuration files to use ArrayFire using CMake.")

af_component(
  COMPONENT licenses
  DISPLAY_NAME "Licenses"
  DESCRIPTION "License files for ArrayFire and its upstream libraries."
  HIDDEN
  REQUIRED)

if(NOT CPACK_GENERATOR MATCHES "DEB")
  af_component(
    COMPONENT common_backend_dependencies
    DISPLAY_NAME "Common Dependencies"
    DESCRIPTION "Libraries commonly required by all ArrayFire backends."
    HIDDEN
    INSTALL_TYPES All Development Runtime)

  af_component(
    COMPONENT cuda_dependencies
    DISPLAY_NAME "CUDA Dependencies"
    DESCRIPTION "Shared libraries required for the CUDA backend."
    HIDDEN
    INSTALL_TYPES All Development Runtime)

endif()

#TODO(pradeep) Remove check after OSX support addition
# Debug symbols in debian installers are created using the DEBINFO property
if(NOT APPLE AND
   NOT CPACK_GENERATOR MATCHES "DEB")
  af_component(
    COMPONENT afoneapi_debug_symbols
    DISPLAY_NAME "oneAPI Debug Symbols"
    DESCRIPTION "Debug symbols for the oneAPI backend."
    GROUP debug
    DISABLED
    INSTALL_TYPES Development)

  af_component(
    COMPONENT afopencl_debug_symbols
    DISPLAY_NAME "OpenCL Debug Symbols"
    DESCRIPTION "Debug symbols for the OpenCL backend."
    GROUP debug
    DISABLED
    INSTALL_TYPES Development)

  af_component(
    COMPONENT afcuda_debug_symbols
    DISPLAY_NAME "CUDA Debug Symbols"
    DESCRIPTION "Debug symbols for CUDA backend backend."
    GROUP debug
    DISABLED
    INSTALL_TYPES Development)

  af_component(
    COMPONENT afcpu_debug_symbols
    DISPLAY_NAME "CPU Debug Symbols"
    DESCRIPTION "Debug symbols for CPU backend backend."
    GROUP debug
    DISABLED
    INSTALL_TYPES Development)

  af_component(
    COMPONENT af_debug_symbols
    DISPLAY_NAME "Unified Debug Symbols"
    DESCRIPTION "Debug symbols for the Unified backend."
    GROUP debug
    DISABLED
    INSTALL_TYPES Development)
endif()

# if (AF_INSTALL_FORGE_DEV)
#   list(APPEND CPACK_COMPONENTS_ALL forge)
#   af_component(
#     COMPONENT forge
#     DISPLAY_NAME "Forge Vizualiation"
#     DESCRIPTION "Visualization Library"
#     INSTALL_TYPES Extra)
# endif ()
#
#set(LIBRARY_NAME ${PROJECT_NAME})
#string(TOLOWER "${LIBRARY_NAME}" APP_LOW_NAME)
#set(SITE_URL "https://arrayfire.com")
#
# set(inst_pkg_name ${APP_LOW_NAME})
# set(inst_pkg_hash "")
# if (WIN32)
#   set(inst_pkg_name ${CPACK_PACKAGE_NAME})
#   set(inst_pkg_hash "-${GIT_COMMIT_HASH}")
# endif ()
#
#set(CPACK_PACKAGE_FILE_NAME "${inst_pkg_name}${inst_pkg_hash}")

# ##
# # IFW CPACK generator
# # Uses Qt installer framework, cross platform installer generator.
# # Uniform installer GUI on all major desktop platforms: Windows, OSX & Linux.
# ##
# set(CPACK_IFW_PACKAGE_TITLE "${CPACK_PACKAGE_NAME}")
# set(CPACK_IFW_PACKAGE_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
# set(CPACK_IFW_PRODUCT_URL "${SITE_URL}")
# set(CPACK_IFW_PACKAGE_ICON "${MY_CPACK_PACKAGE_ICON}")
# set(CPACK_IFW_PACKAGE_WINDOW_ICON "${CMAKE_SOURCE_DIR}/assets/${APP_LOW_NAME}_icon.png")
# set(CPACK_IFW_PACKAGE_WIZARD_DEFAULT_WIDTH 640)
# set(CPACK_IFW_PACKAGE_WIZARD_DEFAULT_HEIGHT 480)
# if (WIN32)
#     set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "@ApplicationsDirX64@/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
# else ()
#     set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "/opt/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
# endif ()
#
# function(get_native_path out_path path)
#   file(TO_NATIVE_PATH ${path} native_path)
#   if (WIN32)
#     string(REPLACE "\\" "\\\\" native_path  ${native_path})
#     set(${out_path} ${native_path} PARENT_SCOPE)
#   else ()
#     set(${out_path} ${path} PARENT_SCOPE)
#   endif ()
# endfunction()
#
# get_native_path(zlib_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/zlib-libpng License.txt")
# get_native_path(boost_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/Boost Software License.txt")
# get_native_path(fimg_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/FreeImage Public License.txt")
# get_native_path(apache_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/Apache-2.0.txt")
# get_native_path(sift_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/OpenSIFT License.txt")
# get_native_path(bsd3_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/BSD 3-Clause.txt")
# get_native_path(issl_lic_path "${CPACK_ArrayFire_SOURCE_DIR}/LICENSES/ISSL License.txt")

#cpack_ifw_configure_component_group(backends)
#cpack_ifw_configure_component_group(cpu-backend)
#cpack_ifw_configure_component_group(cuda-backend)
#cpack_ifw_configure_component_group(opencl-backend)
#if (PACKAGE_MKL_DEPS)
#  cpack_ifw_configure_component(mkl_dependencies)
#endif ()
#if (NOT APPLE)
#  cpack_ifw_configure_component(opencl_dependencies)
#endif ()
#cpack_ifw_configure_component(common_backend_dependencies)
#cpack_ifw_configure_component(cuda_dependencies)
#cpack_ifw_configure_component(cpu)
#cpack_ifw_configure_component(cuda)
#cpack_ifw_configure_component(opencl)
#cpack_ifw_configure_component(unified)
#cpack_ifw_configure_component(headers)
#cpack_ifw_configure_component(cmake)
#cpack_ifw_configure_component(documentation)
#cpack_ifw_configure_component(examples)
#cpack_ifw_configure_component(licenses FORCED_INSTALLATION
#  LICENSES "GLFW" ${zlib_lic_path} "FreeImage" ${fimg_lic_path}
#  "Boost" ${boost_lic_path} "CLBlast, clFFT" ${apache_lic_path} "SIFT" ${sift_lic_path}
#  "BSD3" ${bsd3_lic_path} "Intel MKL" ${issl_lic_path}
#)
#if (AF_INSTALL_FORGE_DEV)
#  cpack_ifw_configure_component(forge)
#endif ()


