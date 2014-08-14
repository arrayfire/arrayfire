#pragma once
#include <complex>

namespace cpu {
    typedef std::complex<float> cfloat;
    typedef std::complex<double> cdouble;
    typedef unsigned int   uint;
    typedef unsigned char uchar;
}

namespace detail = cpu;
