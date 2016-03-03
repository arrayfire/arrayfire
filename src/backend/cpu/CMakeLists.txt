
ADD_DEFINITIONS(-DAF_CPU)

FIND_PACKAGE(CBLAS REQUIRED)

OPTION(BUILD_CPU_ASYNC "Build CPU backend with ASYNC support" ON)

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
ELSE()
    MESSAGE(STATUS "Using CBLAS Library: ${CBLAS_LIBRARIES}")
ENDIF()

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND "${APPLE}")
    ADD_DEFINITIONS(-flax-vector-conversions)
ENDIF()

FIND_PACKAGE(FFTW REQUIRED)
MESSAGE(STATUS "FFTW Found ? ${FFTW_FOUND}")
MESSAGE(STATUS "FFTW Library: ${FFTW_LIBRARIES}")

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
    MESSAGE(STATUS "LAPACK libraries found: ${LAPACK_LIBRARIES}")
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
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend/cpu"
    "${CMAKE_SOURCE_DIR}/src/backend/cpu/threads"
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


IF(DEFINED BLAS_SYM_FILE)
  ADD_LIBRARY(afcpu_static STATIC
              ${cpu_headers}
              ${cpu_sources}
              ${backend_headers}
              ${backend_sources})

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
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/afcpu_static.renamed
      COMMAND objcopy --redefine-syms ${BLAS_SYM_FILE} $<TARGET_FILE:afcpu_static> ${CMAKE_BINARY_DIR}/afcpu_static.renamed
      DEPENDS $<TARGET_FILE:afcpu_static>)
      TARGET_LINK_LIBRARIES(afcpu PUBLIC ${CMAKE_BINARY_DIR}/afcpu_static.renamed)
  ENDIF(APPLE)

ELSE(DEFINED BLAS_SYM_FILE)

  ADD_LIBRARY(afcpu SHARED
              ${cpu_headers}
              ${cpu_sources}
              ${backend_headers}
              ${backend_sources}
              ${c_headers}
              ${c_sources}
              ${cpp_sources})

ENDIF(DEFINED BLAS_SYM_FILE)

TARGET_LINK_LIBRARIES(afcpu
                            PRIVATE ${FreeImage_LIBS}
                            PRIVATE ${CBLAS_LIBRARIES}
                            PRIVATE ${FFTW_LIBRARIES})

IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
  ADD_DEPENDENCIES(afcpu forge)
ENDIF()

IF(LAPACK_FOUND)
   TARGET_LINK_LIBRARIES(afcpu  PRIVATE ${LAPACK_LIBRARIES})
ENDIF()

IF(FORGE_FOUND)
    TARGET_LINK_LIBRARIES(afcpu PRIVATE ${FORGE_LIBRARIES})
ENDIF()

SET_TARGET_PROPERTIES(afcpu PROPERTIES
                      VERSION "${AF_VERSION}"
                      SOVERSION "${AF_VERSION_MAJOR}")

INSTALL(TARGETS afcpu EXPORT CPU DESTINATION "${AF_INSTALL_LIB_DIR}"
        COMPONENT libraries)

IF(APPLE)
    INSTALL(SCRIPT "${CMAKE_MODULE_PATH}/osx_install/InstallTool.cmake")
ENDIF(APPLE)

export(TARGETS afcpu FILE ArrayFireCPU.cmake)
INSTALL(EXPORT CPU DESTINATION "${AF_INSTALL_CMAKE_DIR}"
        COMPONENT cmake
        FILE ArrayFireCPU.cmake)
