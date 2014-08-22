#pragma once
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <complex>

namespace opencl {
    typedef cl_float2   cfloat;
    typedef cl_double2 cdouble;
    typedef cl_uchar     uchar;
    typedef cl_uint       uint;

    template<typename T> static inline T abs(T val)  { return std::abs(val); }

#define INSTANTIATE(T)                          \
    template<> T   abs<T>(T val);               \

    INSTANTIATE(float);
    INSTANTIATE(double);
    INSTANTIATE(int);
    INSTANTIATE(uint);
    INSTANTIATE(char);
    INSTANTIATE(uchar);

#undef INSTANTIATE

    static inline float  abs(cfloat  cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }
    static inline double abs(cdouble cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }
}

namespace detail = opencl;
