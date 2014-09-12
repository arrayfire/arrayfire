#include <err_common.hpp>

#DEfine cpu_NOT_SUPPORTED() do {                       \
        throw SupportError(__func__, __LINE__, "CPU"); \
    } while(0)
