# - Find the NVRTC include directory and libraries
# Modified version of the file found here:
# https://raw.githubusercontent.com/nvidia-compiler-sdk/nvvmir-samples/master/CMakeLists.txt
# CUDA_NVRTC_FOUND
# CUDA_NVRTC_INCLUDE_DIR
# CUDA_NVRTC_LIBRARY

# libNVRTC
IF(NOT DEFINED ENV{CUDA_NVRTC_HOME})
    # If the toolkit path was changed then refind the library
    IF(NOT "${CUDA_NVRTC_HOME}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR}/nvrtc")
        UNSET(CUDA_NVRTC_HOME CACHE)
        UNSET(CUDA_nvrtc_INCLUDE_DIR CACHE)
        UNSET(CUDA_nvrtc_LIBRARY CACHE)
        SET(CUDA_NVRTC_HOME "${CUDA_TOOLKIT_ROOT_DIR}/nvrtc" CACHE INTERNAL "CUDA NVRTC Directory")
    ENDIF()
ELSE()
    SET(CUDA_NVRTC_HOME "$ENV{CUDA_NVRTC_HOME}" CACHE INTERNAL "CUDA NVRTC Directory")
    MESSAGE(STATUS "Using CUDA_NVRTC_HOME: ${CUDA_NVRTC_HOME}")
ENDIF()

FIND_LIBRARY(CUDA_nvrtc_LIBRARY
             NAMES "nvrtc"
             PATHS ${CUDA_NVRTC_HOME} ${CUDA_TOOLKIT_ROOT_DIR}
             PATH_SUFFIXES "lib64" "lib" "lib/x64" "lib/Win32"
             DOC "CUDA NVRTC Library"
            )

FIND_PATH(CUDA_nvrtc_INCLUDE_DIR
          NAMES nvrtc.h
          PATHS ${CUDA_NVRTC_HOME} ${CUDA_TOOLKIT_ROOT_DIR}
          PATH_SUFFIXES "include"
          DOC "CUDA NVRTC Include Directory"
         )

MARK_AS_ADVANCED(
    CUDA_nvrtc_INCLUDE_DIR
    CUDA_nvrtc_LIBRARY)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NVRTC DEFAULT_MSG CUDA_nvrtc_INCLUDE_DIR CUDA_nvrtc_LIBRARY)
