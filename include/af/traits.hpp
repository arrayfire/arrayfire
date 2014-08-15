#pragma once


#ifdef __cplusplus

#include <complex>
#include <af/defines.h>

namespace af {

typedef std::complex<float> af_cfloat;
typedef std::complex<double> af_cdouble;

template<typename T> struct dtype_traits;

template<>
struct dtype_traits<float> {
    enum { af_type = f32 };
    static const char* getName() { return "float"; }
};

template<>
struct dtype_traits<af_cfloat> {
    enum { af_type = c32 };
    static const char* getName() { return "std::complex<float>"; }
};

template<>
struct dtype_traits<double> {
    enum { af_type = f64 };
    static const char* getName() { return "double"; }
};

template<>
struct dtype_traits<af_cdouble> {
    enum { af_type = c64 };
    static const char* getName() { return "std::complex<double>"; }
};

template<>
struct dtype_traits<char> {
    enum { af_type = b8 };
    static const char* getName() { return "char"; }
};

template<>
struct dtype_traits<int> {
    enum { af_type = s32 };
    static const char* getName() { return "int"; }
};

template<>
struct dtype_traits<unsigned> {
    enum { af_type = u32 };
    static const char* getName() { return "unsigned int"; }
};

template<>
struct dtype_traits<unsigned char> {
    enum { af_type = u8 };
    static const char* getName() { return "unsigned char"; }
};

}

#endif
