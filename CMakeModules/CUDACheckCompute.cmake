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

    MESSAGE(STATUS "CUDA Compute Detection Output: ${RUN_OUTPUT_VAR}")
    MESSAGE(STATUS "CUDA Compute Detection Return: ${RUN_RESULT_VAR}")

    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # Check Return Value of main() from RUN_RESULT_VAR
    # RUN_RESULT_VAR is 0 when a GPU is found
    # RUN_RESULT_VAR is 1 when errors occur

    IF(COMPILE_RESULT_VAR AND RUN_RESULT_VAR EQUAL 0)
        MESSAGE(STATUS "CUDA Compute Detection Worked")
        # Convert output into a list of computes
        STRING(REPLACE " " ";" COMPUTES_DETECTED_LIST ${RUN_OUTPUT_VAR})
        SET(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
    ELSE()
        MESSAGE(STATUS "CUDA Compute Detection Failed")
        SET(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
    ENDIF()

ENDIF(CUDA_FOUND)
