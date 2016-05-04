# - Find the NVVM include directory and libraries
# Modified version of the file found here:
# https://raw.githubusercontent.com/nvidia-compiler-sdk/nvvmir-samples/master/CMakeLists.txt
# CUDA_NVVM_FOUND
# CUDA_NVVM_INCLUDE_DIR
# CUDA_NVVM_LIBRARY

# libNVVM
IF(NOT DEFINED ENV{CUDA_NVVM_HOME})
    # If the toolkit path was changed then refind the library
    IF(NOT "${CUDA_NVVM_HOME}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR}/nvvm")
        UNSET(CUDA_NVVM_HOME CACHE)
        UNSET(CUDA_nvvm_INCLUDE_DIR CACHE)
        UNSET(CUDA_nvvm_LIBRARY CACHE)
        SET(CUDA_NVVM_HOME "${CUDA_TOOLKIT_ROOT_DIR}/nvvm" CACHE INTERNAL "CUDA NVVM Directory")
    ENDIF()
ELSE()
  SET(CUDA_NVVM_HOME "$ENV{CUDA_NVVM_HOME}" CACHE INTERNAL "CUDA NVVM Directory")
  MESSAGE(STATUS "Using CUDA_NVVM_HOME: ${CUDA_NVVM_HOME}")
ENDIF()

FIND_LIBRARY(CUDA_nvvm_LIBRARY
             NAMES "nvvm"
             PATHS ${CUDA_NVVM_HOME}
             PATH_SUFFIXES "lib64" "lib" "lib/x64" "lib/Win32"
             DOC "CUDA NVVM Library"
            )

FIND_PATH(CUDA_nvvm_INCLUDE_DIR
          NAMES nvvm.h
          PATHS ${CUDA_NVVM_HOME}
          PATH_SUFFIXES "include"
          DOC "CUDA NVVM Include Directory"
         )

MARK_AS_ADVANCED(
    CUDA_nvvm_INCLUDE_DIR
    CUDA_nvvm_LIBRARY)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NVVM DEFAULT_MSG
    CUDA_nvvm_INCLUDE_DIR CUDA_nvvm_LIBRARY)
