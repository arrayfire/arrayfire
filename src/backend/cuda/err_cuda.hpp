#include <err_common.hpp>

#define CUDA_NOT_SUPPORTED() do {                       \
        throw SupportError(__func__, __LINE__, "CUDA"); \
    } while(0)
