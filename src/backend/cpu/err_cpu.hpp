#include <err_common.hpp>

#define CPU_NOT_SUPPORTED() do {                       \
        throw SupportError(__FILE__, __LINE__, "CPU"); \
    } while(0)
