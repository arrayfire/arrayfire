#pragma once
#include <complex>

namespace cpu {
    typedef std::complex<float>     cfloat;
    typedef std::complex<double>    cdouble;
    typedef unsigned int            uint;
    typedef unsigned char           uchar;

    template<typename T> static inline T abs(T val) { return std::abs(val); }

#define INSTANTIATE(T)                          \
    template<> T   abs<T>(T val);               \

    INSTANTIATE(float);
    INSTANTIATE(double);
    INSTANTIATE(int);
    INSTANTIATE(uint);
    INSTANTIATE(char);
    INSTANTIATE(uchar);

#undef INSTANTIATE

    static inline float  abs(cfloat  cval) { return std::abs(cval); }
    static inline double abs(cdouble cval) { return std::abs(cval); }
}

namespace detail = cpu;
