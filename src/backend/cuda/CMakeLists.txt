CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA 7.0 REQUIRED)

INCLUDE(CLKernelToH)
INCLUDE(FindNVRTC)

MARK_AS_ADVANCED(
    CUDA_BUILD_CUBIN
    CUDA_BUILD_EMULATION
    CUDA_SDK_ROOT_DIR
    CUDA_VERBOSE_BUILD)

# Disables running cuda_compute_check.c when build windows using remote
OPTION(CUDA_COMPUTE_DETECT "Run autodetection of CUDA Architecture" ON)
MARK_AS_ADVANCED(CUDA_COMPUTE_DETECT)

IF(CUDA_COMPUTE_DETECT AND NOT DEFINED COMPUTES_DETECTED_LIST)
    INCLUDE(CUDACheckCompute)
ENDIF()

IF(    CUDA_COMPUTE_20
    OR CUDA_COMPUTE_30
    OR CUDA_COMPUTE_32
    OR CUDA_COMPUTE_35
    OR CUDA_COMPUTE_37
    OR CUDA_COMPUTE_50
    OR CUDA_COMPUTE_52
    OR CUDA_COMPUTE_53
    OR CUDA_COMPUTE_60
    OR CUDA_COMPUTE_61
    OR CUDA_COMPUTE_62
    )
    SET(FALLBACK OFF)
ELSE()
    SET(FALLBACK ON)
ENDIF()

LIST(LENGTH COMPUTES_DETECTED_LIST COMPUTES_LEN)
IF(${COMPUTES_LEN} EQUAL 0 AND ${FALLBACK})
    MESSAGE(STATUS "You can use -DCOMPUTES_DETECTED_LIST=\"AB;XY\" (semicolon separated list of CUDA Compute versions to enable the specified computes")
    MESSAGE(STATUS "Individual compute versions flags are also available under CMake Advance options")
    LIST(APPEND COMPUTES_DETECTED_LIST "30" "50")
    IF(${CUDA_VERSION_MAJOR} GREATER 7) # Enable 60 only if CUDA 8 or greater
        MESSAGE(STATUS "No computes detected. Fall back to 30, 50, 60")
        LIST(APPEND COMPUTES_DETECTED_LIST "60")
    ELSE(${CUDA_VERSION_MAJOR} GREATER 7)
        LIST(APPEND COMPUTES_DETECTED_LIST "20")
    ENDIF(${CUDA_VERSION_MAJOR} GREATER 7)
ENDIF()

LIST(LENGTH COMPUTES_DETECTED_LIST COMPUTES_LEN)
MESSAGE(STATUS "Number of Computes Detected = ${COMPUTES_LEN}")

FOREACH(COMPUTE_DETECTED ${COMPUTES_DETECTED_LIST})
    SET(CUDA_COMPUTE_${COMPUTE_DETECTED} ON CACHE BOOL "" FORCE)
ENDFOREACH()

MACRO(SET_COMPUTE VERSION)
    SET(CUDA_GENERATE_CODE_${VERSION} "-gencode arch=compute_${VERSION},code=sm_${VERSION}")
    SET(CUDA_GENERATE_CODE ${CUDA_GENERATE_CODE} ${CUDA_GENERATE_CODE_${VERSION}})
    LIST(APPEND COMPUTE_VERSIONS "${VERSION}")
    ADD_DEFINITIONS(-DCUDA_COMPUTE_${VERSION})
    MESSAGE(STATUS "Setting Compute ${VERSION} to ON")
ENDMACRO(SET_COMPUTE)

# Iterate over compute versions. Create variables and enable computes if needed
FOREACH(VER 20 30 32 35 37 50 52 53 60 61 62)
    OPTION(CUDA_COMPUTE_${VER} "CUDA Compute Capability ${VER}" OFF)
    MARK_AS_ADVANCED(CUDA_COMPUTE_${VER})
    IF(${CUDA_COMPUTE_${VER}})
        SET_COMPUTE(${VER})
    ENDIF()
ENDFOREACH()

# Error out if Compute 6x is enabled but CUDA version is less than 8
IF(${CUDA_VERSION_MAJOR} LESS 8)
    IF(   CUDA_COMPUTE_60
       OR CUDA_COMPUTE_61
       OR CUDA_COMPUTE_62
      )
        MESSAGE(FATAL_ERROR
                "CUDA Compute 6x was enabled.\
                 CUDA Compute 6x (Pascal) GPUs require CUDA 8 or greater.\
                 Your CUDA Version is ${CUDA_VERSION}."
                )
    ENDIF()
ENDIF(${CUDA_VERSION_MAJOR} LESS 8)

IF(UNIX)
    # GCC 5.3 and above give errors for mempcy from <string.h>
    # This is a (temporary) fix for that
    # This was fixed in CUDA 8.0
    IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.3.0" AND
       ${CUDA_VERSION_MAJOR} LESS 8)
        ADD_DEFINITIONS(-D_FORCE_INLINES)
    ENDIF()

    # Forcing STRICT ANSI should resolve a bunch of issues that NVIDIA seems to face with GCC compilers.
    ADD_DEFINITIONS(-D__STRICT_ANSI__)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fvisibility=hidden)
    IF(${WITH_COVERAGE})
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fprofile-arcs -Xcompiler -ftest-coverage -Xlinker -fprofile-arcs -Xlinker -ftest-coverage")
    ENDIF(${WITH_COVERAGE})
ELSE()
    ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_CUDA)

# CMake 3.2 Adds CUDA_cusolver_LIBRARY variable to FindCUDA
# Older version, use FIND_LIBRARY
IF(CMAKE_VERSION VERSION_LESS 3.2)
    IF(${CUDA_cusolver_LIBRARY} MATCHES " ")
        UNSET(CUDA_cusolver_LIBRARY CACHE)  # When going from higher version to lower version
    ENDIF()

    # Use CUDA_cusolver_DIR to keep track of CUDA Toolkit for which cusolver was found.
    # If the toolkit changed, then find cusolver again
    IF(NOT "${CUDA_cusolver_DIR}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR}")
        UNSET(CUDA_cusolver_DIR CACHE)
        UNSET(CUDA_cusolver_LIBRARY CACHE)
        FIND_LIBRARY (
            CUDA_cusolver_LIBRARY
            NAMES "cusolver"
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES "lib64" "lib/x64" "lib"
            DOC "CUDA cusolver Library"
            NO_DEFAULT_PATH
            )
        SET(CUDA_cusolver_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE INTERNAL "CUDA cusolver Root Directory")
    ENDIF()
    MARK_AS_ADVANCED(CUDA_cusolver_LIBRARY)
    MESSAGE(STATUS "CUDA cusolver library available in CUDA Version ${CUDA_VERSION_STRING}")
ENDIF(CMAKE_VERSION VERSION_LESS 3.2)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/src/backend/cuda"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CUDA_nvrtc_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cub
    )

FILE(GLOB cuda_headers
     "*.hpp"
     "*.h")

FILE(GLOB cuda_sources
    "*.cu"
    "*.cpp"
    "kernel/*.cu")

FILE(GLOB jit_sources
    "JIT/*.hpp")

FILE(GLOB kernel_headers
    "kernel/*.hpp")

LIST(SORT cuda_headers)
LIST(SORT cuda_sources)
LIST(SORT jit_sources)
LIST(SORT kernel_headers)

SOURCE_GROUP(backend\\cuda\\Headers FILES ${cuda_headers})
SOURCE_GROUP(backend\\cuda\\Sources FILES ${cuda_sources})
SOURCE_GROUP(backend\\cuda\\JIT FILES ${jit_sources})
SOURCE_GROUP(backend\\cuda\\kernel\\Headers FILES ${kernel_headers})

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )

FILE(GLOB backend_sources
    "../common/*.cpp"
    "../*.cpp"
    )

LIST(SORT backend_headers)
LIST(SORT backend_sources)

SOURCE_GROUP(backend\\Headers FILES ${backend_headers})
SOURCE_GROUP(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

LIST(SORT c_headers)
LIST(SORT c_sources)

SOURCE_GROUP(api\\c\\Headers FILES ${c_headers})
SOURCE_GROUP(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

LIST(SORT cpp_sources)

SET(jit_kernel_headers
    "kernel_headers")

FILE(GLOB jit_src "kernel/jit.cuh")
CL_KERNEL_TO_H(
    SOURCES ${jit_src}
    VARNAME jit_files
    EXTENSION "hpp"
    OUTPUT_DIR ${jit_kernel_headers}
    TARGETS jit_kernel_targets
    NAMESPACE "cuda"
    )


SOURCE_GROUP(api\\cpp\\Sources FILES ${cpp_sources})

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/CMakeLists.txt")
INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/CMakeLists.txt")

LIST(LENGTH COMPUTE_VERSIONS COMPUTE_COUNT)
IF(${COMPUTE_COUNT} EQUAL 1)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_GENERATE_CODE}")
ELSE()
    # Use -arch sm_30 if CUDA 8 or greater and compute_20 not defined
    IF(CUDA_COMPUTE_20 OR ${CUDA_VERSION_MAJOR} LESS 8)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_20")
    ELSE()
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_30")
    ENDIF()
ENDIF()

# PUSH/POP --keep-device-functions flag. Only available in CUDA 8 or newer
SET(OLD_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
IF(${CUDA_VERSION_MAJOR} GREATER 7) # CUDA 8 or newer
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --keep-device-functions")
ENDIF()
SET(CUDA_NVCC_FLAGS ${OLD_CUDA_NVCC_FLAGS})

IF("${APPLE}")
    ADD_DEFINITIONS(-D__STRICT_ANSI__)
ELSE()
  IF(UNIX)
    IF(${CUDA_VERSION_MAJOR} GREATER 7)
      FIND_PACKAGE(OpenMP)
      IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      ENDIF()
    ENDIF()
  ENDIF()
ENDIF()

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
ENDIF()

## Copied from FindCUDA.cmake
## The target_link_library needs to link with the cuda libraries using
## PRIVATE
macro(MY_CUDA_ADD_LIBRARY cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
      PRIVATE ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()

IF(NOT CUDA_CUDA_LIBRARY)
    MESSAGE(SEND_ERROR "CMake CUDA Variable CUDA_CUDA_LIBRARY Not found.")
    MESSAGE("CUDA Driver Library (libcuda.so/libcuda.dylib/cuda.lib) cannot be found.")
    FIND_FILE(CUDA_CUDA_LIBRARY_STUB
              NAMES "libcuda.so" "libcuda.dylib" "cuda.lib"
              PATHS ${CUDA_TOOLKIT_ROOT_DIR}
              PATH_SUFFIXES "lib64" "lib64/stubs" "lib" "lib/stubs" "lib/x64" "lib/Win32"
              DOC "CUDA Library STUB"
             )
    IF(CUDA_CUDA_LIBRARY_STUB)
        MESSAGE("You can use the library stub available in the CUDA Toolkit: ${CUDA_CUDA_LIBRARY_STUB}")
        MESSAGE("Run the following commands (Linux) to set it up:")
        MESSAGE("ln -s ${CUDA_CUDA_LIBRARY_STUB} /usr/lib/libcuda.so.1")
        MESSAGE("ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so")
    ENDIF()
    MESSAGE(FATAL_ERROR "Ending CMake configuration because of missing CUDA_CUDA_LIBRARY")
ENDIF(NOT CUDA_CUDA_LIBRARY)

SET(CUDA_ADD_LIBRARY_OPTIONS "")
IF(UNIX)
  # These flags enable C++11 and disable invalid offsetof warning
  SET(CUDA_ADD_LIBRARY_OPTIONS "-std=c++11 -Xcudafe \"--diag_suppress=1427\"")
ENDIF(UNIX)


MY_CUDA_ADD_LIBRARY(afcuda SHARED
                ${cuda_headers}
                ${cuda_sources}
                ${jit_sources}
                ${kernel_headers}
                ${backend_headers}
                ${backend_sources}
                ${c_headers}
                ${c_sources}
                ${cpp_sources}
                ${thrust_sort_by_key_sources}
                ${scan_by_key_sources}
                OPTIONS ${CUDA_GENERATE_CODE} ${CUDA_ADD_LIBRARY_OPTIONS})

TARGET_LINK_LIBRARIES(afcuda
    PRIVATE
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${FreeImage_LIBS}
      ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_cusolver_LIBRARY}
      ${CUDA_nvrtc_LIBRARY}
      ${CUDA_CUDA_LIBRARY}
      ${CMAKE_THREAD_LIBS_INIT}
)

ADD_DEPENDENCIES(afcuda ${jit_kernel_targets})
LIST(LENGTH GRAPHICS_DEPENDENCIES GRAPHICS_DEPENDENCIES_LEN)
IF(${GRAPHICS_DEPENDENCIES_LEN} GREATER 0)
    ADD_DEPENDENCIES(afcuda ${GRAPHICS_DEPENDENCIES})
ENDIF(${GRAPHICS_DEPENDENCIES_LEN} GREATER 0)

IF(FORGE_FOUND)
    TARGET_LINK_LIBRARIES(afcuda PRIVATE ${GRAPHICS_LIBRARIES})
ENDIF()

SET_TARGET_PROPERTIES(afcuda PROPERTIES
    VERSION "${AF_VERSION}"
    SOVERSION "${AF_VERSION_MAJOR}")

INSTALL(TARGETS afcuda EXPORT CUDA DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

IF(APPLE)
    INSTALL(SCRIPT "${PROJECT_SOURCE_DIR}/CMakeModules/osx_install/InstallTool.cmake")
ENDIF(APPLE)

export(TARGETS afcuda FILE ArrayFireCUDA.cmake)
INSTALL(EXPORT CUDA DESTINATION "${AF_INSTALL_CMAKE_DIR}"
    COMPONENT cmake
    FILE ArrayFireCUDA.cmake)
