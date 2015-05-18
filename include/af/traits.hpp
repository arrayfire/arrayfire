/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus

#include <complex>
#include <af/defines.h>

namespace af {

template<typename T> struct dtype_traits;

template<>
struct dtype_traits<float> {
    enum { af_type = f32 };
    typedef float base_type;
    static const char* getName() { return "float"; }
};

template<>
struct dtype_traits<af::cfloat> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char* getName() { return "std::complex<float>"; }
};

template<>
struct dtype_traits<double> {
    enum { af_type = f64 };
    typedef double base_type;
    static const char* getName() { return "double"; }
};

template<>
struct dtype_traits<af::cdouble> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char* getName() { return "std::complex<double>"; }
};

template<>
struct dtype_traits<char> {
    enum { af_type = b8 };
    typedef char base_type;
    static const char* getName() { return "char"; }
};

template<>
struct dtype_traits<int> {
    enum { af_type = s32 };
    typedef int base_type;
    static const char* getName() { return "int"; }
};

template<>
struct dtype_traits<unsigned> {
    enum { af_type = u32 };
    typedef unsigned base_type;
    static const char* getName() { return "uint"; }
};

template<>
struct dtype_traits<unsigned char> {
    enum { af_type = u8 };
    typedef unsigned char base_type;
    static const char* getName() { return "uchar"; }
};

template<>
struct dtype_traits<long long> {
    enum { af_type = s64 };
    typedef long long base_type;
    static const char* getName() { return "long"; }
};

template<>
struct dtype_traits<unsigned long long> {
    enum { af_type = u64 };
    typedef unsigned long long base_type;
    static const char* getName() { return "ulong"; }
};

}

#endif
