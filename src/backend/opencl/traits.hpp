#pragma once

#include <af/traits.hpp>

namespace af {

template<>
struct dtype_traits<cl_float2> {
    enum { af_type = c32 };
    static const char* getName() { return "float2"; }
};

template<>
struct dtype_traits<cl_double2> {
    enum { af_type = c64 };
    static const char* getName() { return "double2"; }
};

template<>
struct dtype_traits<size_t> {
    static const char* getName() { return "size_t"; };
};

}

using af::dtype_traits;
