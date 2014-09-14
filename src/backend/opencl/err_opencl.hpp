#include <stdio.h>
#include <err_common.hpp>

#define OPENCL_NOT_SUPPORTED() do {                         \
        throw SupportError(__func__, __LINE__, "OPENCL");   \
    } while(0)

#define CL_TO_AF_ERROR(ERR) do {                        \
        char opencl_err_msg[1024];                      \
        snprintf(opencl_err_msg,                        \
                 sizeof(opencl_err_msg),                \
                 "OpenCL Error: %s when calling %s",    \
                 getErrorMessage(ERR.err()),            \
                 ERR.what());                           \
        AF_ERROR(opencl_err_msg,                        \
                 AF_ERR_INTERNAL);                      \
    } while(0)
