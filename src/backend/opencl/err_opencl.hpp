#include <err_common.hpp>

#define OPENCL_NOT_SUPPORTED() do {                         \
        throw SupportError(__func__, __LINE__, "OPENCL");   \
    } while(0)
