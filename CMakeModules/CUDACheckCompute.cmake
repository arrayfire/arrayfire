#############################
#Sourced from:
#https://raw.githubusercontent.com/jwetzl/CudaLBFGS/master/CheckComputeCapability.cmake
#############################
# Check for GPUs present and their compute capability
# based on http://stackoverflow.com/questions/2285185/easiest-way-to-test-for-existence-of-cuda-capable-gpu-from-cmake/2297877#2297877 (Christopher Bruns)

IF(CUDA_FOUND)
    MESSAGE(STATUS "${CMAKE_MODULE_PATH}/cuda_compute_capability.cpp")
    TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_MODULE_PATH}/cuda_compute_capability.cpp
        CMAKE_FLAGS
        -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
        -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    MESSAGE(STATUS "Output: ${RUN_OUTPUT_VAR}")
    IF(COMPILE_RESULT_VAR)
        # Convert output into a list of computes
        STRING(REPLACE " " ";" COMPUTES_DETECTED_LIST ${RUN_OUTPUT_VAR})
    ELSE()
        MESSAGE(STATUS "didn't compile")
    ENDIF()
    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # RUN_RESULT_VAR is zero when a GPU is found
    IF(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
        MESSAGE(STATUS "CUDA Compute Detection Worked")
        SET(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
    ELSE()
        MESSAGE(STATUS "didn't work")
        SET(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
    ENDIF()
endif()
