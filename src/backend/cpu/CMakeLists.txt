
ADD_DEFINITIONS(-DAF_CPU)

FIND_PACKAGE(CBLAS REQUIRED)

IF(NOT DEFINED BUILD_CPU_ASYNC)
    CMAKE_POLICY(PUSH)
    # https://cmake.org/cmake/help/v3.1/policy/CMP0054.html
    IF("${CMAKE_VERSION}" VERSION_GREATER "3.1" OR "${CMAKE_VERSION}" VERSION_EQUAL "3.1")
        CMAKE_POLICY(SET CMP0054 OLD)
    ENDIF()

    IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.4")
        MESSAGE("Disabling CPU Async as GCC Version ${COMPILER_VERSION} has known issues.")
        MESSAGE("CPU Backend will use Synchronous Calls")
        OPTION(BUILD_CPU_ASYNC "Build CPU backend with ASYNC support" OFF)
    ELSE()
        OPTION(BUILD_CPU_ASYNC "Build CPU backend with ASYNC support" ON)
    ENDIF()

    CMAKE_POLICY(POP)
ENDIF(NOT DEFINED BUILD_CPU_ASYNC)
SET(USE_CPUID ON CACHE BOOL "Build with CPUID integration")
MARK_AS_ADVANCED(USE_CPUID)

if (USE_CPUID)
  ADD_DEFINITIONS(-DUSE_CPUID=1)
ELSE(USE_CPUID)
  ADD_DEFINITIONS(-DUSE_CPUID=0)
ENDIF(USE_CPUID)

IF (NOT ${BUILD_CPU_ASYNC})
    ADD_DEFINITIONS(-DAF_DISABLE_CPU_ASYNC)
ENDIF()

IF(USE_CPU_F77_BLAS)
    MESSAGE("Using F77 BLAS")
    ADD_DEFINITIONS(-DUSE_F77_BLAS)
ENDIF()

IF(USE_CPU_MKL) # Manual MKL Setup
    MESSAGE("CPU Backend Using MKL")
    ADD_DEFINITIONS(-DUSE_MKL)
ELSE(USE_CPU_MKL)
    IF(${MKL_FOUND}) # Automatic MKL Setup from BLAS
        MESSAGE("CPU Backend Using MKL RT")
        ADD_DEFINITIONS(-DUSE_MKL)
    ENDIF()
ENDIF()

IF (NOT CBLAS_LIBRARIES)
    MESSAGE(SEND_ERROR "CBLAS Library not set")
ENDIF()

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND "${APPLE}")
    ADD_DEFINITIONS(-flax-vector-conversions)
ENDIF()

FIND_PACKAGE(FFTW REQUIRED)

IF(APPLE)
    FIND_PACKAGE(LAPACKE QUIET) # For finding MKL
    IF(NOT LAPACK_FOUND)
        # UNSET THE VARIABLES FROM LAPACKE
        UNSET(LAPACKE_LIB CACHE)
        UNSET(LAPACK_LIB CACHE)
        UNSET(LAPACKE_INCLUDES CACHE)
        UNSET(LAPACKE_ROOT_DIR CACHE)
        FIND_PACKAGE(LAPACK)
    ENDIF()
ELSE(APPLE) # Linux and Windows
    FIND_PACKAGE(LAPACKE)
ENDIF(APPLE)

IF(NOT LAPACK_FOUND)
    MESSAGE(WARNING "LAPACK not found. Functionality will be disabled")
ELSE(NOT LAPACK_FOUND)
    ADD_DEFINITIONS(-DWITH_CPU_LINEAR_ALGEBRA)
ENDIF()

IF(NOT UNIX)
    ADD_DEFINITIONS(-DAFDLL)
ENDIF()

SET(THREADS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/threads")
IF(EXISTS "${THREADS_SRC_DIR}" AND IS_DIRECTORY "${THREADS_SRC_DIR}"
        AND EXISTS "${THREADS_SRC_DIR}/LICENSE")
    # threads submodule has been initialized
    # Nothing to do
ELSE()
    MESSAGE(STATUS "threads submodule unavailable. Updating submodules.")
    EXECUTE_PROCESS(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${PROJECT_SOURCE_DIR}/src/backend/cpu"
    "${PROJECT_SOURCE_DIR}/src/backend/cpu/threads"
    ${FFTW_INCLUDES}
    ${CBLAS_INCLUDE_DIR}
    )

IF(LAPACK_FOUND)
    INCLUDE_DIRECTORIES(${LAPACK_INCLUDE_DIR})
ENDIF()

FILE(GLOB cpu_headers
    "*.hpp"
    "*.h")

FILE(GLOB cpu_sources
    "*.cpp")

LIST(SORT cpu_headers)
LIST(SORT cpu_sources)

source_group(backend\\cpu\\Headers FILES ${cpu_headers})
source_group(backend\\cpu\\Sources FILES ${cpu_sources})

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

source_group(backend\\Headers FILES ${backend_headers})
source_group(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

LIST(SORT c_headers)
LIST(SORT c_sources)

source_group(api\\c\\Headers FILES ${c_headers})
source_group(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

LIST(SORT cpp_sources)

source_group(api\\cpp\\Sources FILES ${cpp_sources})

# OS Definitions
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ELSE(${UNIX}) #Windows
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/CMakeLists.txt")

IF(DEFINED BLAS_SYM_FILE)
  ADD_LIBRARY(afcpu_static STATIC
              ${cpu_headers}
              ${cpu_sources}
              ${backend_headers}
              ${backend_sources}
              ${SORT_BY_KEY_OBJECTS})

  ADD_LIBRARY(afcpu SHARED
              ${c_headers}
              ${c_sources}
              ${cpp_sources})

  IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
    ADD_DEPENDENCIES(afcpu_static forge)
  ENDIF()

  IF(APPLE)
    SET_TARGET_PROPERTIES(afcpu_static
        PROPERTIES LINK_FLAGS -Wl,-exported_symbols_list,${BLAS_SYM_FILE})
    TARGET_LINK_LIBRARIES(afcpu PUBLIC $<TARGET_FILE:afcpu_static>)
  ELSE(APPLE)
    add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/afcpu_static.renamed
      COMMAND objcopy --redefine-syms ${BLAS_SYM_FILE} $<TARGET_FILE:afcpu_static> ${PROJECT_BINARY_DIR}/afcpu_static.renamed
      DEPENDS $<TARGET_FILE:afcpu_static>)
      TARGET_LINK_LIBRARIES(afcpu PUBLIC ${PROJECT_BINARY_DIR}/afcpu_static.renamed)
  ENDIF(APPLE)

ELSE(DEFINED BLAS_SYM_FILE)

ADD_LIBRARY(afcpu SHARED
            ${cpu_headers}
            ${cpu_sources}
            ${backend_headers}
            ${backend_sources}
            ${c_headers}
            ${c_sources}
            ${cpp_sources}
            ${SORT_BY_KEY_OBJECTS})

ENDIF(DEFINED BLAS_SYM_FILE)

TARGET_LINK_LIBRARIES(afcpu
                            PRIVATE ${CBLAS_LIBRARIES}
                            PRIVATE ${FFTW_LIBRARIES}
                            PRIVATE ${FreeImage_LIBS}
                     )

IF(LAPACK_FOUND)
   TARGET_LINK_LIBRARIES(afcpu  PRIVATE ${LAPACK_LIBRARIES})
ENDIF()

LIST(LENGTH GRAPHICS_DEPENDENCIES GRAPHICS_DEPENDENCIES_LEN)
IF(${GRAPHICS_DEPENDENCIES_LEN} GREATER 0)
    ADD_DEPENDENCIES(afcpu ${GRAPHICS_DEPENDENCIES})
ENDIF(${GRAPHICS_DEPENDENCIES_LEN} GREATER 0)

IF(FORGE_FOUND)
   TARGET_LINK_LIBRARIES(afcpu PRIVATE ${GRAPHICS_LIBRARIES})
ENDIF()

SET_TARGET_PROPERTIES(afcpu PROPERTIES
                      VERSION "${AF_VERSION}"
                      SOVERSION "${AF_VERSION_MAJOR}")

INSTALL(TARGETS afcpu EXPORT CPU DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

IF(APPLE)
    INSTALL(SCRIPT "${PROJECT_SOURCE_DIR}/CMakeModules/osx_install/InstallTool.cmake")
ENDIF(APPLE)

export(TARGETS afcpu FILE ArrayFireCPU.cmake)
INSTALL(EXPORT CPU DESTINATION "${AF_INSTALL_CMAKE_DIR}"
        COMPONENT cmake
        FILE ArrayFireCPU.cmake)
