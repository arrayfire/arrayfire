# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.


# determine the compiler to use for C++ programs
# NOTE, a generator may set CMAKE_SYCL_COMPILER before
# loading this file to force a compiler.
# use environment variable SYCL first if defined by user, next use
# the cmake variable CMAKE_GENERATOR_SYCL which can be defined by a generator
# as a default compiler
# If the internal cmake variable _CMAKE_TOOLCHAIN_PREFIX is set, this is used
# as prefix for the tools (e.g. arm-elf-g++, arm-elf-ar etc.)
#
# Sets the following variables:
#   CMAKE_SYCL_COMPILER
#   CMAKE_COMPILER_IS_GNUSYCL
#   CMAKE_AR
#   CMAKE_RANLIB
#
# If not already set before, it also sets
#   _CMAKE_TOOLCHAIN_PREFIX

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_ROOT})
include(CMakeDetermineCompiler)

# Load system-specific compiler preferences for this language.
#include(Platform/${CMAKE_SYSTEM_NAME}-Determine-SYCL OPTIONAL)
#include(Platform/${CMAKE_SYSTEM_NAME}-SYCL OPTIONAL)
if(NOT CMAKE_SYCL_COMPILER_NAMES)
  set(CMAKE_SYCL_COMPILER_NAMES icpx)
endif()

if(${CMAKE_GENERATOR} MATCHES "Visual Studio")
elseif("${CMAKE_GENERATOR}" MATCHES "Green Hills MULTI")
elseif("${CMAKE_GENERATOR}" MATCHES "Xcode")
  set(CMAKE_SYCL_COMPILER_XCODE_TYPE sourcecode.cpp.cpp)
  _cmake_find_compiler_path(SYCL)
else()
  if(NOT CMAKE_SYCL_COMPILER)
    set(CMAKE_SYCL_COMPILER_INIT NOTFOUND)

    # prefer the environment variable SYCL
    if(NOT $ENV{SYCL} STREQUAL "")
      get_filename_component(CMAKE_SYCL_COMPILER_INIT $ENV{SYCL} PROGRAM PROGRAM_ARGS CMAKE_SYCL_FLAGS_ENV_INIT)
      if(CMAKE_SYCL_FLAGS_ENV_INIT)
        set(CMAKE_SYCL_COMPILER_ARG1 "${CMAKE_SYCL_FLAGS_ENV_INIT}" CACHE STRING "Arguments to SYCL compiler")
      endif()
      if(NOT EXISTS ${CMAKE_SYCL_COMPILER_INIT})
        message(FATAL_ERROR "Could not find compiler set in environment variable SYCL:\n$ENV{SYCL}.\n${CMAKE_SYCL_COMPILER_INIT}")
      endif()
    endif()

    # next prefer the generator specified compiler
    if(CMAKE_GENERATOR_SYCL)
      if(NOT CMAKE_SYCL_COMPILER_INIT)
        set(CMAKE_SYCL_COMPILER_INIT ${CMAKE_GENERATOR_SYCL})
      endif()
    endif()

    # finally list compilers to try
    if(NOT CMAKE_SYCL_COMPILER_INIT)
      set(CMAKE_SYCL_COMPILER_LIST icpx icx)
      if(NOT CMAKE_HOST_WIN32)
        # FIXME(#24314): Add support for the GNU-like icpx compiler driver
        # on Windows, first introduced by Intel oneAPI 2023.0.
        list(APPEND CMAKE_SYCL_COMPILER_LIST icpx)
      endif()
    endif()

    _cmake_find_compiler(SYCL)
  else()
    _cmake_find_compiler_path(SYCL)
  endif()
  mark_as_advanced(CMAKE_SYCL_COMPILER)

  # Each entry in this list is a set of extra flags to try
  # adding to the compile line to see if it helps produce
  # a valid identification file.
  set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS_FIRST)
  set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS
    "-fsycl"
    # Try compiling to an object file only.
    "-c"
    # IAR does not detect language automatically
    "--c++"
    "--ec++"

    # ARMClang need target options
    "--target=arm-arm-none-eabi -mcpu=cortex-m3"

    # MSVC needs at least one include directory for __has_include to function,
    # but custom toolchains may run MSVC with no INCLUDE env var and no -I flags.
    # Also avoid linking so this works with no LIB env var.
    "-c -I__does_not_exist__"
    )
endif()

if(CMAKE_SYCL_COMPILER_TARGET)
  set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS_FIRST "-c --target=${CMAKE_SYCL_COMPILER_TARGET}")
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_SYCL_COMPILER_ID_RUN)
  set(CMAKE_SYCL_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_SYCL_COMPILER_ID)
  set(CMAKE_SYCL_PLATFORM_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in
    CMAKE_SYCL_COMPILER_ID_PLATFORM_CONTENT)

  # The IAR compiler produces weird output.
  # See https://gitlab.kitware.com/cmake/cmake/-/issues/10176#note_153591
  list(APPEND CMAKE_SYCL_COMPILER_ID_VENDORS IAR)
  set(CMAKE_SYCL_COMPILER_ID_VENDOR_FLAGS_IAR )
  set(CMAKE_SYCL_COMPILER_ID_VENDOR_REGEX_IAR "IAR .+ Compiler")

  # Match the link line from xcodebuild output of the form
  #  Ld ...
  #      ...
  #      /path/to/cc ...CompilerIdSYCL/...
  # to extract the compiler front-end for the language.
  set(CMAKE_SYCL_COMPILER_ID_TOOL_MATCH_REGEX "\nLd[^\n]*(\n[ \t]+[^\n]*)*\n[ \t]+([^ \t\r\n]+)[^\r\n]*-o[^\r\n]*CompilerIdSYCL/(\\./)?(CompilerIdSYCL.(framework|xctest|build/[^ \t\r\n]+)/)?CompilerIdSYCL[ \t\n\\\"]")
  set(CMAKE_SYCL_COMPILER_ID_TOOL_MATCH_INDEX 2)

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  set(SYCLFLAGS "-fsycl -Werror")
  CMAKE_DETERMINE_COMPILER_ID(SYCL SYCLFLAGS CMakeSYCLCompilerId.cpp)

  _cmake_find_compiler_sysroot(SYCL)

  # Set old compiler and platform id variables.
  if(CMAKE_SYCL_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_COMPILER_IS_GNUSYCL 1)
  endif()
else()
  if(NOT DEFINED CMAKE_SYCL_COMPILER_FRONTEND_VARIANT)
    # Some toolchain files set our internal CMAKE_SYCL_COMPILER_ID_RUN
    # variable but are not aware of CMAKE_SYCL_COMPILER_FRONTEND_VARIANT.
    # They pre-date our support for the GNU-like variant targeting the
    # MSVC ABI so we do not consider that here.
    if(CMAKE_SYCL_COMPILER_ID STREQUAL "Clang"
      OR "x${CMAKE_SYCL_COMPILER_ID}" STREQUAL "xIntelLLVM")
      if("x${CMAKE_SYCL_SIMULATE_ID}" STREQUAL "xMSVC")
        set(CMAKE_SYCL_COMPILER_FRONTEND_VARIANT "MSVC")
      else()
        set(CMAKE_SYCL_COMPILER_FRONTEND_VARIANT "GNU")
      endif()
    else()
      set(CMAKE_SYCL_COMPILER_FRONTEND_VARIANT "")
    endif()
  endif()
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_SYCL_COMPILER}" PATH)
endif ()

# if we have a g++ cross compiler, they have usually some prefix, like
# e.g. powerpc-linux-g++, arm-elf-g++ or i586-mingw32msvc-g++ , optionally
# with a 3-component version number at the end (e.g. arm-eabi-gcc-4.5.2).
# The other tools of the toolchain usually have the same prefix
# NAME_WE cannot be used since then this test will fail for names like
# "arm-unknown-nto-qnx6.3.0-gcc.exe", where BASENAME would be
# "arm-unknown-nto-qnx6" instead of the correct "arm-unknown-nto-qnx6.3.0-"


if (NOT _CMAKE_TOOLCHAIN_PREFIX)

  if("${CMAKE_SYCL_COMPILER_ID}" MATCHES "GNU|Clang|QCC|LCC")
    get_filename_component(COMPILER_BASENAME "${CMAKE_SYCL_COMPILER}" NAME)
    if (COMPILER_BASENAME MATCHES "^(.+-)?(clang\\+\\+|[gc]\\+\\+|clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
      set(_CMAKE_TOOLCHAIN_SUFFIX ${CMAKE_MATCH_3})
      set(_CMAKE_COMPILER_SUFFIX ${CMAKE_MATCH_5})
    elseif("${CMAKE_SYCL_COMPILER_ID}" MATCHES "Clang")
      if(CMAKE_SYCL_COMPILER_TARGET)
        set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_SYCL_COMPILER_TARGET}-)
      endif()
    elseif(COMPILER_BASENAME MATCHES "QCC(\\.exe)?$")
      if(CMAKE_SYCL_COMPILER_TARGET MATCHES "gcc_nto([a-z0-9]+_[0-9]+|[^_le]+)(le)")
        set(_CMAKE_TOOLCHAIN_PREFIX nto${CMAKE_MATCH_1}-)
      endif()
    endif ()

    # if "llvm-" is part of the prefix, remove it, since llvm doesn't have its own binutils
    # but uses the regular ar, objcopy, etc. (instead of llvm-objcopy etc.)
    if ("${_CMAKE_TOOLCHAIN_PREFIX}" MATCHES "(.+-)?llvm-$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
    endif ()
  elseif("${CMAKE_SYCL_COMPILER_ID}" MATCHES "TI")
    # TI compilers are named e.g. cl6x, cl470 or armcl.exe
    get_filename_component(COMPILER_BASENAME "${CMAKE_SYCL_COMPILER}" NAME)
    if (COMPILER_BASENAME MATCHES "^(.+)?cl([^.]+)?(\\.exe)?$")
      set(_CMAKE_TOOLCHAIN_PREFIX "${CMAKE_MATCH_1}")
      set(_CMAKE_TOOLCHAIN_SUFFIX "${CMAKE_MATCH_2}")
    endif ()

  endif()

endif ()

set(_CMAKE_PROCESSING_LANGUAGE "SYCL")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_SYCL_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(CMAKE_SYCL_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_SYCL_COMPILER_SYSROOT
    "set(CMAKE_SYCL_COMPILER_SYSROOT \"${CMAKE_SYCL_COMPILER_SYSROOT}\")\n"
    "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_SYCL_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_SYCL_COMPILER_SYSROOT "")
endif()

if(CMAKE_SYCL_COMPILER_ARCHITECTURE_ID)
  set(_SET_CMAKE_SYCL_COMPILER_ARCHITECTURE_ID
    "set(CMAKE_SYCL_COMPILER_ARCHITECTURE_ID ${CMAKE_SYCL_COMPILER_ARCHITECTURE_ID})")
else()
  set(_SET_CMAKE_SYCL_COMPILER_ARCHITECTURE_ID "")
endif()

if(MSVC_SYCL_ARCHITECTURE_ID)
  set(SET_MSVC_SYCL_ARCHITECTURE_ID
    "set(MSVC_SYCL_ARCHITECTURE_ID ${MSVC_SYCL_ARCHITECTURE_ID})")
endif()

if(CMAKE_SYCL_XCODE_ARCHS)
  set(SET_CMAKE_XCODE_ARCHS
    "set(CMAKE_XCODE_ARCHS \"${CMAKE_SYCL_XCODE_ARCHS}\")")
endif()

# configure all variables set in this file
configure_file(${ArrayFire_SOURCE_DIR}/CMakeModules/CMakeSYCLCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake
  @ONLY
  )

set(CMAKE_SYCL_COMPILER_ENV_VAR "SYCL")
