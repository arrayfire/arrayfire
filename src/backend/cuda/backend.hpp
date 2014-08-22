#pragma once
#include <cuComplex.h>

namespace cuda {
    typedef cuFloatComplex   cfloat;
    typedef cuDoubleComplex cdouble;
    typedef unsigned int   uint;
    typedef unsigned char uchar;
}

namespace detail = cuda;
