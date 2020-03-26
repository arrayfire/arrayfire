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

namespace common {
class half;
}

#ifdef __CUDACC_RTC__

using dim_t = long long;

#else  //__CUDACC_RTC__

#include <af/traits.hpp>

#endif  //__CUDACC_RTC__

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
const char *shortname(bool caps = false) {
    return caps ? "Q" : "q";
}
template<>
const char *shortname<float>(bool caps) {
    return caps ? "S" : "s";
}
template<>
const char *shortname<double>(bool caps) {
    return caps ? "D" : "d";
}
template<>
const char *shortname<cfloat>(bool caps) {
    return caps ? "C" : "c";
}
template<>
const char *shortname<cdouble>(bool caps) {
    return caps ? "Z" : "z";
}
template<>
const char *shortname<int>(bool caps) {
    return caps ? "I" : "i";
}
template<>
const char *shortname<uint>(bool caps) {
    return caps ? "U" : "u";
}
template<>
const char *shortname<char>(bool caps) {
    return caps ? "J" : "j";
}
template<>
const char *shortname<uchar>(bool caps) {
    return caps ? "V" : "v";
}
template<>
const char *shortname<intl>(bool caps) {
    return caps ? "X" : "x";
}
template<>
const char *shortname<uintl>(bool caps) {
    return caps ? "Y" : "y";
}
template<>
const char *shortname<short>(bool caps) {
    return caps ? "P" : "p";
}
template<>
const char *shortname<ushort>(bool caps) {
    return caps ? "Q" : "q";
}
template<>
const char *shortname<common::half>(bool caps) {
    return caps ? "H" : "h";
}

template<typename T>
const char *getFullName();

#define SPECIALIZE(T)              \
    template<>                     \
    const char *getFullName<T>() { \
        return #T;                 \
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
const char *getFullName<common::half>() {
    return "half";
}
#undef SPECIALIZE
}  // namespace
#endif  //__CUDACC_RTC__

//#ifndef __CUDACC_RTC__
}  // namespace cuda
//#endif  //__CUDACC_RTC__

namespace common {
template<typename T>
class kernel_type;
}

namespace common {
template<>
struct kernel_type<common::half> {
    using data = common::half;

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

#if defined(NVCC) || defined(__CUDACC_RTC__)
    using native  = __half;
#else
    using native = common::half;
#endif

#endif  // __CUDA_ARCH__
};
}  // namespace common
