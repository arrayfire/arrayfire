
ADD_DEFINITIONS(-DAF_CPU)

FIND_PACKAGE(CBLAS REQUIRED)

IF(USE_CPU_F77_BLAS)
    MESSAGE("Using F77 BLAS")
    ADD_DEFINITIONS(-DUSE_F77_BLAS)
ENDIF()

IF(USE_CPU_MKL)
    MESSAGE("Using MKL")
    ADD_DEFINITIONS(-DUSE_MKL)
ENDIF()

IF (NOT CBLAS_LIBRARIES)
    MESSAGE(SEND_ERROR "CBLAS Library not set")
ELSE()
    MESSAGE(STATUS "Using CBLAS Library: ${CBLAS_LIBRARIES}")
ENDIF()

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND "${APPLE}")
    ADD_DEFINITIONS(-flax-vector-conversions)
ENDIF()

IF(${MKL_FOUND})
    ADD_DEFINITIONS(-DUSE_MKL)
ENDIF()

FIND_PACKAGE(FFTW REQUIRED)
MESSAGE(STATUS "FFTW Found ? ${FFTW_FOUND}")
MESSAGE(STATUS "FFTW Library: ${FFTW_LIBRARIES}")

IF(APPLE)
    FIND_PACKAGE(LAPACK)
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

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend/cpu"
    ${FFTW_INCLUDES}
    ${CBLAS_INCLUDE_DIR}
    ${LAPACK_INCLUDE_DIR}
    )

FILE(GLOB cpu_headers
    "*.hpp"
    "*.h")

FILE(GLOB cpu_sources
    "*.cpp")

source_group(backend\\cpu\\Headers FILES ${cpu_headers})
source_group(backend\\cpu\\Sources FILES ${cpu_sources})

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )

FILE(GLOB backend_sources
    "../*.cpp"
    )

source_group(backend\\Headers FILES ${backend_headers})
source_group(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

source_group(api\\c\\Headers FILES ${c_headers})
source_group(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

source_group(api\\cpp\\Sources FILES ${cpp_sources})

# OS Definitions
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ELSE(${UNIX}) #Windows
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

ADD_LIBRARY(afcpu SHARED
           ${cpu_headers}
           ${cpu_sources}
           ${backend_headers}
           ${backend_sources}
           ${c_headers}
           ${c_sources}
           ${cpp_sources})

TARGET_LINK_LIBRARIES(afcpu PRIVATE ${FreeImage_LIBS}
                            PRIVATE ${CBLAS_LIBRARIES}
                            PRIVATE ${FFTW_LIBRARIES})

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
