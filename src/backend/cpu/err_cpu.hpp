#include <err_common.hpp>

#define CPU_NOT_SUPPORTED() do {                       \
        throw SupportError(__func__, __LINE__, "CPU"); \
    } while(0)
