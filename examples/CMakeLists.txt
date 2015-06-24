CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(arrayfire-examples)

# Find CUDA and OpenCL
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(OpenCL QUIET)

# If the examples are not being built at the same time as ArrayFire,
# we need to first find the ArrayFire library
if(TARGET afcpu OR TARGET afcuda OR TARGET afopencl)
    SET(ArrayFire_CPU_FOUND False)
    SET(ArrayFire_CUDA_FOUND False)
    SET(ArrayFire_OpenCL_FOUND False)
    SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
    IF(NOT EXISTS "${ASSETS_DIR}/LICENSE")
        MESSAGE(WARNING "Arrayfire assets are not available. Assets will not be installed.")
        MESSAGE("Did you miss the --recursive option when cloning?")
        MESSAGE("Run the following commands to correct this:")
        MESSAGE("git submodule init")
        MESSAGE("git submodule update")
        MESSAGE("git submodule foreach git pull origin master")
    ENDIF()
else()
    FIND_PACKAGE(ArrayFire REQUIRED)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

    SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
endif()

IF(WIN32)
    # Deprecated Errors are Warning 4996 on VS2013.
    # https://msdn.microsoft.com/en-us/library/ttcz0bys.aspx
    SET(CMAKE_CXX_FLAGS "/we4996")
    SET(CMAKE_C_FLAGS   "/we4996")
ELSE(WIN32)
    SET(CMAKE_CXX_FLAGS "-Werror=deprecated-declarations")
    SET(CMAKE_C_FLAGS   "-Werror=deprecated-declarations")
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

    FOREACH(FILE ${FILES})
        GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
        GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
        GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

        BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${BACKEND_NAME} "${BACKEND_LIBRARIES}" "${OTHER_LIBRARIES}" ${DIR_NAME})
    ENDFOREACH()
ENDMACRO()

# Collect the source
FILE(GLOB FILES "*/*.cpp")
ADD_DEFINITIONS("-DASSETS_DIR=\"${ASSETS_DIR}\"")

# Next we build each example using every backend.
if(${ArrayFire_CPU_FOUND})  # variable defined by FIND(ArrayFire ...)
  MESSAGE(STATUS "EXAMPLES: CPU backend is ON.")
  BUILD_ALL("${FILES}" cpu ${ArrayFire_CPU_LIBRARIES} "")
elseif(TARGET afcpu)        # variable defined by the ArrayFire build tree
  MESSAGE(STATUS "EXAMPLES: CPU backend is ON.")
  BUILD_ALL("${FILES}" cpu afcpu "")
else()
  MESSAGE(STATUS "EXAMPLES: CPU backend is OFF. afcpu was not found.")
endif()

if (${CUDA_FOUND})
  if(${ArrayFire_CUDA_FOUND})  # variable defined by FIND(ArrayFire ...)
    FIND_LIBRARY( CUDA_NVVM_LIBRARY
      NAMES "nvvm"
      PATH_SUFFIXES "nvvm/lib64" "nvvm/lib"
      PATHS ${CUDA_TOOLKIT_ROOT_DIR}
      DOC "CUDA NVVM Library"
      )
    MESSAGE(STATUS "EXAMPLES: CUDA backend is ON.")
    BUILD_ALL("${FILES}" cuda ${ArrayFire_CUDA_LIBRARIES} "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES};${CUDA_cusolver_LIBRARY};${CUDA_CUFFT_LIBRARIES};${CUDA_NVVM_LIBRARY};${CUDA_CUDA_LIBRARY}")
  elseif(TARGET afcuda)        # variable defined by the ArrayFire build tree
    MESSAGE(STATUS "EXAMPLES: CUDA backend is ON.")
    BUILD_ALL("${FILES}" cuda afcuda "")
  else()
    MESSAGE(STATUS "EXAMPLES: CUDA backend is OFF. afcuda was not found")
  endif()
else()
  MESSAGE(STATUS "EXAMPLES: CUDA backend is OFF. CUDA was not found")
endif()

if (${OpenCL_FOUND})
  if(${ArrayFire_OpenCL_FOUND})  # variable defined by FIND(ArrayFire ...)
    MESSAGE(STATUS "EXAMPLES: OPENCL backend is ON.")
    BUILD_ALL("${FILES}" opencl ${ArrayFire_OpenCL_LIBRARIES} "${OpenCL_LIBRARIES}")
  elseif(TARGET afopencl)        # variable defined by the ArrayFire build tree
    MESSAGE(STATUS "EXAMPLES: OPENCL backend is ON.")
    BUILD_ALL("${FILES}" opencl afopencl ${OpenCL_LIBRARIES})
  else()
    MESSAGE(STATUS "EXAMPLES: OPENCL backend is OFF. afopencl was not found")
  endif()
else()
  MESSAGE(STATUS "EXAMPLES: OPENCL backend is OFF. OPENCL was not found")
endif()

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION "${AF_INSTALL_EXAMPLE_DIR}"
    COMPONENT examples)

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../assets/examples"
    DESTINATION "${AF_INSTALL_EXAMPLE_DIR}/examples/assets/"
)
