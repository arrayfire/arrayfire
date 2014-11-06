
FIND_PACKAGE(CBLAS REQUIRED)
MESSAGE(STATUS "BLAS Library: ${BLAS_LIBRARIES}")

FIND_PACKAGE(FFTW REQUIRED)
MESSAGE(STATUS "FFTW Found ? ${FFTW_FOUND}")
MESSAGE(STATUS "FFTW Library: ${FFTW_LIBRARIES}")

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    ${CBLAS_INCLUDE_DIR}
    ${FFTW_INCLUDES}
    )

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )
FILE(GLOB backend_sources
    "../*.cpp"
    )
FILE(GLOB cpp_sources
    "../../array/*.cpp"
    )
source_group(Headers FILES ${backend_headers})
source_group(Sources FILES ${backend_sources})
source_group(CPP\ Sources FILES ${cpp_sources})

FILE(GLOB cpu_headers
	"*.hpp"
    "*.h")
FILE(GLOB cpu_sources
    "*.cpp")
source_group(cpu\\Headers FILES ${cpu_headers})
source_group(cpu\\Sources FILES ${cpu_sources})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ADD_LIBRARY(afcpu SHARED 	${cpp_sources}
							${backend_headers}
							${backend_sources}
							${cpu_headers}
							${cpu_sources})

TARGET_LINK_LIBRARIES(afcpu ${lib_deps} ${CBLAS_LIBRARIES} ${FFTW_LIBRARIES})
