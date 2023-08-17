# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# make sure default modules are accesible
list(APPEND CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules)
message(${CMAKE_MODULE_PATH})

set(CMAKE_SYCL_COMPILER_ID IntelLLVM)

# This file sets the basic flags for the C++ language in CMake.
# It also loads the available platform file for the system-compiler
# if it exists.
# It also loads a system - compiler - processor (or target hardware)
# specific file, which is mainly useful for crosscompiling and embedded systems.

include(CMakeLanguageInformation)

# some compilers use different extensions (e.g. sdcc uses .rel)
# so set the extension here first so it can be overridden by the compiler specific file
if(UNIX)
  set(CMAKE_SYCL_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_SYCL_OUTPUT_EXTENSION .obj)
endif()

set(_INCLUDED_FILE 0)

# Load compiler-specific information.
if(CMAKE_SYCL_COMPILER_ID)
  #include(Compiler/${CMAKE_SYCL_COMPILER_ID}-CXX OPTIONAL)
endif()

set(CMAKE_BASE_NAME)
get_filename_component(CMAKE_BASE_NAME "${CMAKE_SYCL_COMPILER}" NAME_WE)
# since the gnu compiler has several names force g++
if(CMAKE_COMPILER_IS_GNUSYCL)
  set(CMAKE_BASE_NAME g++)
endif()

include(Compiler/${CMAKE_SYCL_COMPILER_ID} OPTIONAL)
__compiler_intel_llvm(SYCL)

if("x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC")
  set(CMAKE_SYCL_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -TP)
  set(CMAKE_SYCL_CLANG_TIDY_DRIVER_MODE "cl")
  set(CMAKE_SYCL_INCLUDE_WHAT_YOU_USE_DRIVER_MODE "cl")
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
      AND CMAKE_DEPFILE_FLAGS_SYCL)
    set(CMAKE_SYCL_DEPENDS_USE_COMPILER TRUE)
  endif()
else()
  set(CMAKE_SYCL_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
      AND CMAKE_DEPFILE_FLAGS_SYCL)
    # dependencies are computed by the compiler itself
    set(CMAKE_SYCL_DEPFILE_FORMAT gcc)
    set(CMAKE_SYCL_DEPENDS_USE_COMPILER TRUE)
  endif()

  set(CMAKE_SYCL_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fvisibility-inlines-hidden")

  string(APPEND CMAKE_SYCL_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
  string(APPEND CMAKE_SYCL_FLAGS_RELEASE_INIT " -DNDEBUG")
  string(APPEND CMAKE_SYCL_FLAGS_RELWITHDEBINFO_INIT " -DNDEBUG")
endif()

set(CMAKE_SYCL98_STANDARD__HAS_FULL_SUPPORT ON)
set(CMAKE_SYCL11_STANDARD__HAS_FULL_SUPPORT ON)
set(CMAKE_SYCL14_STANDARD__HAS_FULL_SUPPORT ON)

if(NOT "x${CMAKE_SYCL_SIMULATE_ID}" STREQUAL "xMSVC")
  set(CMAKE_SYCL98_STANDARD_COMPILE_OPTION  "-std=c++98")
  set(CMAKE_SYCL98_EXTENSION_COMPILE_OPTION "-std=gnu++98")

  set(CMAKE_SYCL11_STANDARD_COMPILE_OPTION  "-std=c++11")
  set(CMAKE_SYCL11_EXTENSION_COMPILE_OPTION "-std=gnu++11")

  set(CMAKE_SYCL14_STANDARD_COMPILE_OPTION  "-std=c++14")
  set(CMAKE_SYCL14_EXTENSION_COMPILE_OPTION "-std=gnu++14")

  set(CMAKE_SYCL17_STANDARD_COMPILE_OPTION  "-std=c++17")
  set(CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=gnu++17")

  set(CMAKE_SYCL20_STANDARD_COMPILE_OPTION  "-std=c++20")
  set(CMAKE_SYCL20_EXTENSION_COMPILE_OPTION "-std=gnu++20")

  set(CMAKE_SYCL23_STANDARD_COMPILE_OPTION  "-std=c++2b")
  set(CMAKE_SYCL23_EXTENSION_COMPILE_OPTION "-std=gnu++2b")
else()
  set(CMAKE_SYCL98_STANDARD_COMPILE_OPTION  "")
  set(CMAKE_SYCL98_EXTENSION_COMPILE_OPTION "")

  set(CMAKE_SYCL11_STANDARD_COMPILE_OPTION  "")
  set(CMAKE_SYCL11_EXTENSION_COMPILE_OPTION "")

  set(CMAKE_SYCL14_STANDARD_COMPILE_OPTION  "-Qstd:c++14")
  set(CMAKE_SYCL14_EXTENSION_COMPILE_OPTION "-Qstd:c++14")

  set(CMAKE_SYCL17_STANDARD_COMPILE_OPTION  "-Qstd:c++17")
  set(CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-Qstd:c++17")

  set(CMAKE_SYCL20_STANDARD_COMPILE_OPTION  "-Qstd:c++20")
  set(CMAKE_SYCL20_EXTENSION_COMPILE_OPTION "-Qstd:c++20")

  set(CMAKE_SYCL23_STANDARD_COMPILE_OPTION  "-Qstd:c++2b")
  set(CMAKE_SYCL23_EXTENSION_COMPILE_OPTION "-Qstd:c++2b")
endif()

include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_SYCL_COMPILER_ID} OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)

if(WIN32)
  set(_COMPILE_CXX " /TP")
  __windows_compiler_intel(SYCL)
elseif(UNIX AND NOT APPLE)
  __linux_compiler_intel_llvm(SYCL)
  # This should be -isystem but icpx throws an error on Ubuntu
  # when you include /usr/include as a system header
  set(CMAKE_INCLUDE_SYSTEM_FLAG_SYCL "-I ")
else()
  __apple_compiler_intel_llvm(SYCL)
endif()

# We specify the compiler information in the system file for some
# platforms, but this language may not have been enabled when the file
# was first included.  Include it again to get the language info.
# Remove this when all compiler info is removed from system files.
if (NOT _INCLUDED_FILE)
  include(Platform/${CMAKE_SYSTEM_NAME} OPTIONAL)
endif ()

if(CMAKE_SYCL_SIZEOF_DATA_PTR)
  foreach(f ${CMAKE_SYCL_ABI_FILES})
    include(${f})
  endforeach()
  unset(CMAKE_SYCL_ABI_FILES)
endif()

# This should be included before the _INIT variables are
# used to initialize the cache.  Since the rule variables
# have if blocks on them, users can still define them here.
# But, it should still be after the platform file so changes can
# be made to those values.

if(CMAKE_USER_MAKE_RULES_OVERRIDE)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE "${_override}")
endif()

if(CMAKE_USER_MAKE_RULES_OVERRIDE_SYCL)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_SYCL} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_SYCL "${_override}")
endif()


# Create a set of shared library variable specific to C++
# For 90% of the systems, these are the same flags as the C versions
# so if these are not set just copy the flags from the c version
if(NOT CMAKE_SHARED_LIBRARY_CREATE_SYCL_FLAGS)
  set(CMAKE_SHARED_LIBRARY_CREATE_SYCL_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS})
endif()

if(NOT CMAKE_SYCL_COMPILE_OPTIONS_PIC)
  set(CMAKE_SYCL_COMPILE_OPTIONS_PIC ${CMAKE_CXX_COMPILE_OPTIONS_PIC})
endif()

if(NOT CMAKE_SYCL_COMPILE_OPTIONS_PIE)
  set(CMAKE_SYCL_COMPILE_OPTIONS_PIE ${CMAKE_CXX_COMPILE_OPTIONS_PIE})
endif()
if(NOT CMAKE_SYCL_LINK_OPTIONS_PIE)
  set(CMAKE_SYCL_LINK_OPTIONS_PIE ${CMAKE_CXX_LINK_OPTIONS_PIE})
endif()
if(NOT CMAKE_SYCL_LINK_OPTIONS_NO_PIE)
  set(CMAKE_SYCL_LINK_OPTIONS_NO_PIE ${CMAKE_CXX_LINK_OPTIONS_NO_PIE})
endif()

if(NOT CMAKE_SYCL_COMPILE_OPTIONS_DLL)
  set(CMAKE_SYCL_COMPILE_OPTIONS_DLL ${CMAKE_CXX_COMPILE_OPTIONS_DLL})
endif()

if(NOT CMAKE_SHARED_LIBRARY_SYCL_FLAGS)
  set(CMAKE_SHARED_LIBRARY_SYCL_FLAGS ${CMAKE_SHARED_LIBRARY_CXX_FLAGS})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_SYCL_FLAGS)
  set(CMAKE_SHARED_LIBRARY_LINK_SYCL_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG_SEP)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RPATH_LINK_SYCL_FLAG)
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_SYCL_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})
endif()

if(NOT DEFINED CMAKE_EXE_EXPORTS_SYCL_FLAG)
  set(CMAKE_EXE_EXPORTS_SYCL_FLAG ${CMAKE_EXE_EXPORTS_CXX_FLAG})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_SONAME_SYCL_FLAG)
  set(CMAKE_SHARED_LIBRARY_SONAME_SYCL_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG})
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_SYCL_FLAG)
  set(CMAKE_EXECUTABLE_RUNTIME_SYCL_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG})
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_SYCL_FLAG_SEP)
  set(CMAKE_EXECUTABLE_RUNTIME_SYCL_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_SYCL_FLAG_SEP})
endif()

if(NOT CMAKE_EXECUTABLE_RPATH_LINK_SYCL_FLAG)
  set(CMAKE_EXECUTABLE_RPATH_LINK_SYCL_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_SYCL_FLAG})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_SYCL_WITH_RUNTIME_PATH)
  set(CMAKE_SHARED_LIBRARY_LINK_SYCL_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_CXX_WITH_RUNTIME_PATH})
endif()

if(NOT CMAKE_INCLUDE_FLAG_SYCL)
  set(CMAKE_INCLUDE_FLAG_SYCL ${CMAKE_INCLUDE_FLAG_C})
endif()

# for most systems a module is the same as a shared library
# so unless the variable CMAKE_MODULE_EXISTS is set just
# copy the values from the LIBRARY variables
if(NOT CMAKE_MODULE_EXISTS)
  set(CMAKE_SHARED_MODULE_SYCL_FLAGS ${CMAKE_SHARED_LIBRARY_SYCL_FLAGS})
  set(CMAKE_SHARED_MODULE_CREATE_SYCL_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_SYCL_FLAGS})
endif()

# repeat for modules
if(NOT CMAKE_SHARED_MODULE_CREATE_SYCL_FLAGS)
  set(CMAKE_SHARED_MODULE_CREATE_SYCL_FLAGS ${CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS})
endif()

if(NOT CMAKE_SHARED_MODULE_SYCL_FLAGS)
  set(CMAKE_SHARED_MODULE_SYCL_FLAGS ${CMAKE_SHARED_MODULE_CXX_FLAGS})
endif()

# Initialize SYCL link type selection flags from C versions.
foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
  if(NOT CMAKE_${type}_LINK_STATIC_SYCL_FLAGS)
    set(CMAKE_${type}_LINK_STATIC_SYCL_FLAGS
      ${CMAKE_${type}_LINK_STATIC_CXX_FLAGS})
  endif()
  if(NOT CMAKE_${type}_LINK_DYNAMIC_SYCL_FLAGS)
    set(CMAKE_${type}_LINK_DYNAMIC_SYCL_FLAGS
      ${CMAKE_${type}_LINK_DYNAMIC_CXX_FLAGS})
  endif()
endforeach()

if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  if(NOT DEFINED CMAKE_SYCL_LINK_WHAT_YOU_USE_FLAG)
    set(CMAKE_SYCL_LINK_WHAT_YOU_USE_FLAG "LINKER:--no-as-needed")
  endif()
  if(NOT DEFINED CMAKE_LINK_WHAT_YOU_USE_CHECK)
    set(CMAKE_LINK_WHAT_YOU_USE_CHECK ldd -u -r)
  endif()
endif()

# add the flags to the cache based
# on the initial values computed in the platform/*.cmake files
# use _INIT variables so that this only happens the first time
# and you can set these flags in the cmake cache
set(CMAKE_SYCL_FLAGS_INIT "-fsycl $ENV{SYCLFLAGS} ${CMAKE_SYCL_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_SYCL_FLAGS "Flags used by the SYCL compiler")

if(CMAKE_SYCL_STANDARD_LIBRARIES_INIT)
  set(CMAKE_SYCL_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all C++ applications.")
  mark_as_advanced(CMAKE_SYCL_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_SYCL_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_SYCL_COMPILER_LAUNCHER})
  set(CMAKE_SYCL_COMPILER_LAUNCHER "$ENV{CMAKE_SYCL_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for SYCL.")
endif()

if(NOT CMAKE_SYCL_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_SYCL_LINKER_LAUNCHER})
  set(CMAKE_SYCL_LINKER_LAUNCHER "$ENV{CMAKE_SYCL_LINKER_LAUNCHER}"
    CACHE STRING "Linker launcher for SYCL.")
endif()

include(CMakeCommonLanguageInclude)

# now define the following rules:
# CMAKE_SYCL_CREATE_SHARED_LIBRARY
# CMAKE_SYCL_CREATE_SHARED_MODULE
# CMAKE_SYCL_COMPILE_OBJECT
# CMAKE_SYCL_LINK_EXECUTABLE

# variables supplied by the generator at use time
# <TARGET>
# <TARGET_BASE> the target without the suffix
# <OBJECTS>
# <OBJECT>
# <LINK_LIBRARIES>
# <FLAGS>
# <LINK_FLAGS>

# SYCL compiler information
# <CMAKE_SYCL_COMPILER>
# <CMAKE_SHARED_LIBRARY_CREATE_SYCL_FLAGS>
# <CMAKE_SYCL_SHARED_MODULE_CREATE_FLAGS>
# <CMAKE_SYCL_LINK_FLAGS>

# Static library tools
# <CMAKE_AR>
# <CMAKE_RANLIB>

# create a shared C++ library
if(NOT CMAKE_SYCL_CREATE_SHARED_LIBRARY)
  set(CMAKE_SYCL_CREATE_SHARED_LIBRARY
      "<CMAKE_SYCL_COMPILER> <CMAKE_SHARED_LIBRARY_SYCL_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_SYCL_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create a c++ shared module copy the shared library rule by default
if(NOT CMAKE_SYCL_CREATE_SHARED_MODULE)
  set(CMAKE_SYCL_CREATE_SHARED_MODULE ${CMAKE_SYCL_CREATE_SHARED_LIBRARY})
endif()


# Create a static archive incrementally for large object file counts.
# If CMAKE_SYCL_CREATE_STATIC_LIBRARY is set it will override these.
if(NOT DEFINED CMAKE_SYCL_ARCHIVE_CREATE)
  set(CMAKE_SYCL_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_SYCL_ARCHIVE_APPEND)
  set(CMAKE_SYCL_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_SYCL_ARCHIVE_FINISH)
  set(CMAKE_SYCL_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile a C++ file into an object file
if(NOT CMAKE_SYCL_COMPILE_OBJECT)
  set(CMAKE_SYCL_COMPILE_OBJECT
    "<CMAKE_SYCL_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_SYCL_LINK_EXECUTABLE)
  set(CMAKE_SYCL_LINK_EXECUTABLE
    "<CMAKE_SYCL_COMPILER> <FLAGS> <CMAKE_SYCL_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()


mark_as_advanced(
CMAKE_VERBOSE_MAKEFILE
)

set(CMAKE_SYCL_INFORMATION_LOADED 1)
