/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/kernel_type.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>

namespace arrayfire {
namespace common {
class half;
}  // namespace common
}  // namespace arrayfire

#ifdef __CUDACC_RTC__

using dim_t = long long;

#else  //__CUDACC_RTC__

#include <af/traits.hpp>

#endif  //__CUDACC_RTC__

namespace arrayfire {
namespace cuda {

using cdouble = cuDoubleComplex;
using cfloat  = cuFloatComplex;
using intl    = long long;
using uchar   = unsigned char;
using uint    = unsigned int;
using uintl   = unsigned long long;
using ushort  = unsigned short;
using ulong   = unsigned long long;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

#ifndef __CUDACC_RTC__
namespace {
template<typename T>
inline const char *shortname(bool caps = false) {
    return caps ? "Q" : "q";
}
template<>
inline const char *shortname<float>(bool caps) {
    return caps ? "S" : "s";
}
template<>
inline const char *shortname<double>(bool caps) {
    return caps ? "D" : "d";
}
template<>
inline const char *shortname<cfloat>(bool caps) {
    return caps ? "C" : "c";
}
template<>
inline const char *shortname<cdouble>(bool caps) {
    return caps ? "Z" : "z";
}
template<>
inline const char *shortname<int>(bool caps) {
    return caps ? "I" : "i";
}
template<>
inline const char *shortname<uint>(bool caps) {
    return caps ? "U" : "u";
}
template<>
inline const char *shortname<char>(bool caps) {
    return caps ? "J" : "j";
}
template<>
inline const char *shortname<uchar>(bool caps) {
    return caps ? "V" : "v";
}
template<>
inline const char *shortname<intl>(bool caps) {
    return caps ? "X" : "x";
}
template<>
inline const char *shortname<uintl>(bool caps) {
    return caps ? "Y" : "y";
}
template<>
inline const char *shortname<short>(bool caps) {
    return caps ? "P" : "p";
}
template<>
inline const char *shortname<ushort>(bool caps) {
    return caps ? "Q" : "q";
}
template<>
inline const char *shortname<arrayfire::common::half>(bool caps) {
    return caps ? "H" : "h";
}

template<typename T>
inline const char *getFullName();

#define SPECIALIZE(T)                     \
    template<>                            \
    inline const char *getFullName<T>() { \
        return #T;                        \
    }

SPECIALIZE(float)
SPECIALIZE(double)
SPECIALIZE(cfloat)
SPECIALIZE(cdouble)
SPECIALIZE(char)
SPECIALIZE(unsigned char)
SPECIALIZE(short)
SPECIALIZE(unsigned short)
SPECIALIZE(int)
SPECIALIZE(unsigned int)
SPECIALIZE(unsigned long long)
SPECIALIZE(long long)

template<>
inline const char *getFullName<common::half>() {
    return "half";
}
#undef SPECIALIZE
}  // namespace
#endif  //__CUDACC_RTC__

}  // namespace cuda

namespace common {

template<typename T>
struct kernel_type;

template<>
struct kernel_type<arrayfire::common::half> {
    using data = arrayfire::common::half;

#ifdef __CUDA_ARCH__

    // These are the types within a kernel
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    using compute = __half;
#else
    using compute = float;
#endif
    using native = compute;

#else  // __CUDA_ARCH__

    // outside of a cuda kernel use float
    using compute = float;

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
    using native  = __half;
#else
    using native = common::half;
#endif

#endif  // __CUDA_ARCH__
};
}  // namespace common
}  // namespace arrayfire
