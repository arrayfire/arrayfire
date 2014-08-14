#pragma once

#include <af/traits.hpp>
#include <cuComplex.h>

namespace af {

template<>
struct dtype_traits<cuFloatComplex> {
    enum { af_type = c32 };
    static const char* getName() { return "cuFloatComplex"; }
};

template<>
struct dtype_traits<cuDoubleComplex> {
    enum { af_type = c64 };
    static const char* getName() { return "cuDoubleComplex"; }
};

}

using af::dtype_traits;
