#pragma once
#include <cuComplex.h>
#include <complex>

#ifndef __DH__
#define __DH__
#endif

namespace cuda {
    typedef cuFloatComplex   cfloat;
    typedef cuDoubleComplex cdouble;
    typedef unsigned int   uint;
    typedef unsigned char uchar;

    template<typename T> static inline __DH__ T abs(T val)  { return std::abs(val); }

#define INSTANTIATE(T)                          \
    template<> T   abs<T>(T val);               \

    INSTANTIATE(float);
    INSTANTIATE(double);
    INSTANTIATE(int);
    INSTANTIATE(uint);
    INSTANTIATE(char);
    INSTANTIATE(uchar);

#undef INSTANTIATE

    static inline __DH__ float  abs(cfloat  cval) { return cuCabsf(cval); }
    static inline __DH__ double abs(cdouble cval) { return cuCabs (cval); }
}

namespace detail = cuda;
