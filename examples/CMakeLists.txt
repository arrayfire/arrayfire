CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(arrayfire-examples)

# If the examples are not being built at the same time as ArrayFire,
# we need to first find the ArrayFire library
if(NOT (TARGET afcpu OR TARGET afcuda OR TARGET afopencl))
    FIND_PACKAGE(ArrayFire REQUIRED)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
endif()

# A macro to build an ArrayFire example
# For most uses only FIND_PACKAGE(ArrayFire REQUIRED), ADD_EXECUTABLE(...)
# and TARGET_LINK_LIBRARIES(... ${ARRAYFIRE_LIBRARIES}) are needed
MACRO(BUILD_EXAMPLE EXAMPLE_NAME EXAMPLE_SOURCE BACKEND_NAME BACKEND_LIBRARIES)
    
    ADD_EXECUTABLE(${EXAMPLE_NAME}-${BACKEND_NAME} ${EXAMPLE_SOURCE})
    TARGET_LINK_LIBRARIES(${EXAMPLE_NAME}-${BACKEND_NAME} 
        ${BACKEND_LIBRARIES} )

ENDMACRO()

# Collect the source
FILE(GLOB FILES "*/*.cpp")
SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
ADD_DEFINITIONS("-DASSETS_DIR=\"\\\"${ASSETS_DIR}\\\"\"")

FOREACH(FILE ${FILES})
    GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
    GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
    GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

    # Next we build each example using every backend.
    if(${ArrayFire_CPU_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cpu ${AF_CPU_LIBRARIES})
    elseif(TARGET afcpu)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cpu afcpu)
    endif()

    if(${ArrayFire_CUDA_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cuda ${AF_CUDA_LIBRARIES})
    elseif(TARGET afcuda)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cuda afcuda)
    endif()
    
    if(${ArrayFire_OPENCL_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} opencl ${AF_OPENCL_LIBRARIES})
    elseif(TARGET afopencl)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} opencl afopencl)
    endif()
ENDFOREACH()

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" 
    DESTINATION share/doc/arrayfire/ 
    COMPONENT examples)
