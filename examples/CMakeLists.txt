CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(arrayfire-examples)

# If the examples are not being built at the same time as ArrayFire,
# we need to first find the ArrayFire library
if(TARGET afcpu OR TARGET afcuda OR TARGET afopencl)
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

# A macro to build an ArrayFire example
# For most uses only FIND_PACKAGE(ArrayFire REQUIRED), ADD_EXECUTABLE(...)
# and TARGET_LINK_LIBRARIES(... ${ARRAYFIRE_LIBRARIES}) are needed
MACRO(BUILD_EXAMPLE EXAMPLE_NAME EXAMPLE_SOURCE BACKEND_NAME BACKEND_LIBRARIES)

    ADD_EXECUTABLE(example_${EXAMPLE_NAME}_${BACKEND_NAME} ${EXAMPLE_SOURCE})
    TARGET_LINK_LIBRARIES(example_${EXAMPLE_NAME}_${BACKEND_NAME}
        ${BACKEND_LIBRARIES} )
    SET_TARGET_PROPERTIES(example_${EXAMPLE_NAME}_${BACKEND_NAME}
        PROPERTIES
        OUTPUT_NAME ${EXAMPLE_NAME}_${BACKEND_NAME}
        RUNTIME_OUTPUT_DIRECTORY ${DIR_NAME})

ENDMACRO()

# Collect the source
FILE(GLOB FILES "*/*.cpp")
ADD_DEFINITIONS("-DASSETS_DIR=\"${ASSETS_DIR}\"")

FOREACH(FILE ${FILES})
    GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
    GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
    GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

    # Next we build each example using every backend.
    if(${ArrayFire_CPU_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cpu ${ArrayFire_CPU_LIBRARIES})
    elseif(TARGET afcpu)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cpu afcpu)
    endif()

    if(${ArrayFire_CUDA_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cuda ${ArrayFire_CUDA_LIBRARIES})
    elseif(TARGET afcuda)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} cuda afcuda)
    endif()

    if(${ArrayFire_OpenCL_FOUND})  # variable defined by FIND(ArrayFire ...)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} opencl ${ArrayFire_OpenCL_LIBRARIES})
    elseif(TARGET afopencl)        # variable defined by the ArrayFire build tree
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} opencl afopencl)
    endif()
ENDFOREACH()

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION share/doc/arrayfire/
    COMPONENT examples)

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../assets/examples"
    DESTINATION share/doc/arrayfire/examples/assets/
)
