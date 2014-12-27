
FIND_PACKAGE(CBLAS REQUIRED)
MESSAGE(STATUS "BLAS Library: ${CBLAS_LIBRARIES}")

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND "${APPLE}")
    ADD_DEFINITIONS(-flax-vector-conversions)
ENDIF()

FIND_PACKAGE(FFTW REQUIRED)
MESSAGE(STATUS "FFTW Found ? ${FFTW_FOUND}")
MESSAGE(STATUS "FFTW Library: ${FFTW_LIBRARIES}")

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend/cpu"
    ${FFTW_INCLUDES}
	${CBLAS_INCLUDE_DIR}
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

TARGET_LINK_LIBRARIES(afcpu ${lib_deps} ${CBLAS_LIBRARIES} ${FFTW_LIBRARIES})

INSTALL(TARGETS afcpu DESTINATION lib)
