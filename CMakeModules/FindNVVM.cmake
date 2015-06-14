# - Find the NVVM include directory and libraries
# Modified version of the file found here:
# https://raw.githubusercontent.com/nvidia-compiler-sdk/nvvmir-samples/master/CMakeLists.txt

# libNVVM
if(NOT DEFINED ENV{LIBNVVM_HOME})
  set(LIBNVVM_HOME "${CUDA_TOOLKIT_ROOT_DIR}/nvvm")
else()
  set(LIBNVVM_HOME "$ENV{LIBNVVM_HOME}")
endif()
message(STATUS "Using LIBNVVM_HOME: ${LIBNVVM_HOME}")

IF(${CUDA_VERSION_MAJOR} LESS 7)
	SET(NVVM_DLL_VERSION 20_0)
ELSE(${CUDA_VERSION_MAJOR} LESS 7)
	SET(NVVM_DLL_VERSION 30_0)
ENDIF(${CUDA_VERSION_MAJOR} LESS 7)

if (CMAKE_SIZEOF_VOID_P STREQUAL "8")
  if (WIN32)
    set (CUDA_LIB_SEARCH_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    set (NVVM_DLL_NAME nvvm64_${NVVM_DLL_VERSION}.dll)
  else ()
    set (CUDA_LIB_SEARCH_PATH "")
  endif()
else()
  if (WIN32)
    set (CUDA_LIB_SEARCH_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32")
    set (NVVM_DLL_NAME nvvm32_${NVVM_DLL_VERSION}.dll)
  else()
    set (CUDA_LIB_SEARCH_PATH "")
  endif()
endif()

### Find libNVVM
# The directory structure for nvvm is a bit complex.
# On Windows:
#   32-bit -- nvvm/lib/Win32
#   64-bit -- nvvm/lib/x64
# On Linux:
#   32-bit -- nvvm/lib
#   64-bit -- nvvm/lib64
# On Mac:
#   Universal -- nvvm/lib
if (CMAKE_SIZEOF_VOID_P STREQUAL "8")
  if (WIN32)
    set (LIB_ARCH_SUFFIX "/x64")
  elseif (APPLE)
    set (LIB_ARCH_SUFFIX "")
  else ()
    set (LIB_ARCH_SUFFIX "64")
  endif()
else()
  if (WIN32)
    set (LIB_ARCH_SUFFIX "/Win32")
  else()
    set (LIB_ARCH_SUFFIX "")
  endif()
endif()

find_library(NVVM_LIB nvvm PATHS "${LIBNVVM_HOME}/lib${LIB_ARCH_SUFFIX}")
find_file(NVVM_H nvvm.h PATHS "${LIBNVVM_HOME}/include")

if(NVVM_H)
  get_filename_component(CUDA_NVVM_INCLUDE_DIR ${NVVM_H} PATH)
else()
  message(FATAL_ERROR "Unable to find nvvm.h")
endif()

set(CUDA_NVVM_LIBRARIES ${NVVM_LIB})
