CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ArrayFire-Examples)

# Find CUDA and OpenCL
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(OpenCL QUIET)

# If the examples are not being built at the same time as ArrayFire,
# we need to first find the ArrayFire library
IF(TARGET afcpu OR TARGET afcuda OR TARGET afopencl OR TARGET af)
    SET(ArrayFire_CPU_FOUND False)
    SET(ArrayFire_CUDA_FOUND False)
    SET(ArrayFire_OpenCL_FOUND False)
    SET(ArrayFire_Unified_FOUND False)
    SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
    IF(NOT EXISTS "${ASSETS_DIR}/LICENSE")
        MESSAGE(STATUS "Assests submodule unavailable. Updating submodules.")
        EXECUTE_PROCESS(
            COMMAND git submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_QUIET
        )
    ENDIF()
ELSE()
    SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
    FIND_PACKAGE(ArrayFire REQUIRED)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

    SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
ENDIF()

IF(WIN32)
    # Deprecated Errors are Warning 4996 on VS2013.
    # https://msdn.microsoft.com/en-us/library/ttcz0bys.aspx
    IF(MSVC)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /we4996")
        SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /we4996")
    ENDIF(MSVC)
ELSE(WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=deprecated-declarations")
    SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Werror=deprecated-declarations")
ENDIF(WIN32)

# A macro to build an ArrayFire example
# For most uses only FIND_PACKAGE(ArrayFire REQUIRED), ADD_EXECUTABLE(...)
# and TARGET_LINK_LIBRARIES(... ${ARRAYFIRE_LIBRARIES}) are needed
MACRO(BUILD_EXAMPLE EXAMPLE_NAME EXAMPLE_SOURCE BACKEND_NAME BACKEND_LIBRARIES OTHER_LIBRARIES OUT_DIR_NAME)
    ADD_EXECUTABLE(example_${EXAMPLE_NAME}_${BACKEND_NAME} ${EXAMPLE_SOURCE})
    TARGET_LINK_LIBRARIES(example_${EXAMPLE_NAME}_${BACKEND_NAME}
        ${BACKEND_LIBRARIES} ${OTHER_LIBRARIES})
    SET_TARGET_PROPERTIES(example_${EXAMPLE_NAME}_${BACKEND_NAME}
        PROPERTIES
        OUTPUT_NAME ${EXAMPLE_NAME}_${BACKEND_NAME}
        RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR_NAME}
        FOLDER "Examples/${BACKEND_NAME}")
ENDMACRO()

# A macro to build a list of files
# For most uses only FIND_PACKAGE(ArrayFire REQUIRED), ADD_EXECUTABLE(...)
# and TARGET_LINK_LIBRARIES(... ${ARRAYFIRE_LIBRARIES}) are needed
MACRO(BUILD_ALL FILES BACKEND_NAME BACKEND_LIBRARIES OTHER_LIBRARIES)

    STRING(TOUPPER ${BACKEND_NAME} BACKEND_NAME_UPPER)
    MESSAGE(STATUS "EXAMPLES: ${BACKEND_NAME_UPPER} backend is ${BUILD_${BACKEND_NAME_UPPER}}.")
    IF(${BUILD_${BACKEND_NAME_UPPER}})
        FOREACH(FILE ${FILES})
            GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
            GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
            GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

            BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${BACKEND_NAME} "${BACKEND_LIBRARIES}" "${OTHER_LIBRARIES}" ${DIR_NAME})
        ENDFOREACH()
    ENDIF()
ENDMACRO()

# Collect the source
FILE(GLOB FILES "*/*.cpp")
LIST(SORT FILES)
ADD_DEFINITIONS("-DASSETS_DIR=\"${ASSETS_DIR}\"")

# Next we build each example using every backend.
IF(${ArrayFire_CPU_FOUND})  # variable defined by FIND(ArrayFire ...)
    OPTION(BUILD_CPU "Build ArrayFire Examples for CPU backend" ON)
    BUILD_ALL("${FILES}" cpu ${ArrayFire_CPU_LIBRARIES} "")
ELSEIF(TARGET afcpu)        # variable defined by the ArrayFire build tree
    BUILD_ALL("${FILES}" cpu afcpu "")
ELSE()
    MESSAGE(STATUS "EXAMPLES: CPU backend is OFF. afcpu was not found.")
ENDIF()

# Next we build each example using every backend.
IF(${ArrayFire_Unified_FOUND})  # variable defined by FIND(ArrayFire ...)
    OPTION(BUILD_UNIFIED "Build ArrayFire Examples for Unified backend" ON)
    BUILD_ALL("${FILES}" unified ${ArrayFire_Unified_LIBRARIES} "${CMAKE_DL_LIBS}")
ELSEIF(TARGET af)        # variable defined by the ArrayFire build tree
    BUILD_ALL("${FILES}" unified af "${CMAKE_DL_LIBS}")
ELSE()
    MESSAGE(STATUS "EXAMPLES: UNIFIED backend is OFF. af was not found.")
ENDIF()

IF (${CUDA_FOUND})
    IF(${ArrayFire_CUDA_FOUND})  # variable defined by FIND(ArrayFire ...)
        FIND_LIBRARY( CUDA_NVVM_LIBRARY
          NAMES "nvvm"
          PATH_SUFFIXES "nvvm/lib64" "nvvm/lib"
          PATHS ${CUDA_TOOLKIT_ROOT_DIR}
          DOC "CUDA NVVM Library"
          )
        MARK_AS_ADVANCED(CUDA_NVVM_LIBRARY)
        OPTION(BUILD_CUDA "Build ArrayFire Examples for CUDA backend" ON)
        BUILD_ALL("${FILES}" cuda ${ArrayFire_CUDA_LIBRARIES} "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES};${CUDA_cusolver_LIBRARY};${CUDA_CUFFT_LIBRARIES};${CUDA_NVVM_LIBRARY};${CUDA_CUDA_LIBRARY}")
    ELSEIF(TARGET afcuda)        # variable defined by the ArrayFire build tree
        BUILD_ALL("${FILES}" cuda afcuda "")
    ELSE()
        MESSAGE(STATUS "EXAMPLES: CUDA backend is OFF. afcuda was not found")
    ENDIF()
ELSE()
    MESSAGE(STATUS "EXAMPLES: CUDA backend is OFF. CUDA was not found")
ENDIF()

IF (${OpenCL_FOUND})
    IF(${ArrayFire_OpenCL_FOUND})  # variable defined by FIND(ArrayFire ...)
        OPTION(BUILD_OPENCL "Build ArrayFire Examples for OpenCL backend" ON)
        BUILD_ALL("${FILES}" opencl ${ArrayFire_OpenCL_LIBRARIES} "${OpenCL_LIBRARIES}")
    ELSEIF(TARGET afopencl)        # variable defined by the ArrayFire build tree
        BUILD_ALL("${FILES}" opencl afopencl "${OpenCL_LIBRARIES}")
    ELSE()
        MESSAGE(STATUS "EXAMPLES: OpenCL backend is OFF. afopencl was not found")
    ENDIF()
ELSE()
    MESSAGE(STATUS "EXAMPLES: OpenCL backend is OFF. OpenCL was not found")
ENDIF()
