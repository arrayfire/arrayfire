#include <af/exception.h>

#define AF_THROW(fn) do {                               \
        af_err __err = fn;                              \
        if (__err == AF_SUCCESS) break;                 \
        throw af::exception(__FILE__, __LINE__, __err); \
    } while(0)
