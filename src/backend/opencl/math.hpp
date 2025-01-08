/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <common/half.hpp>
#include <af/defines.h>

#include <backend.hpp>
#include <types.hpp>

#include <algorithm>
#include <complex>
#include <climits>
#include <limits>

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#else
/* Other */
#endif

namespace arrayfire {
namespace opencl {

template<typename T>
static inline T abs(T val) {
    return std::abs(val);
}
template<typename T>
static inline T min(T lhs, T rhs) {
    return std::min(lhs, rhs);
}
template<typename T>
static inline T max(T lhs, T rhs) {
    return std::max(lhs, rhs);
}

static inline float abs(cfloat cval) {
    return std::sqrt(cval.s[0] * cval.s[0] + cval.s[1] * cval.s[1]);
}
static inline double abs(cdouble cval) {
    return std::sqrt(cval.s[0] * cval.s[0] + cval.s[1] * cval.s[1]);
}

template<typename T>
static inline T division(T lhs, double rhs) {
    return lhs / rhs;
}
cfloat division(cfloat lhs, double rhs);
cdouble division(cdouble lhs, double rhs);

template<>
inline cfloat max<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
inline cdouble max<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
inline cfloat min<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<>
inline cdouble min<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<typename T>
static T scalar(double val) {
    return (T)(val);
}

template<>
inline cfloat scalar<cfloat>(double val) {
    cfloat cval;
    cval.s[0] = (float)val;
    cval.s[1] = 0;
    return cval;
}

template<>
inline cdouble scalar<cdouble>(double val) {
    cdouble cval;
    cval.s[0] = val;
    cval.s[1] = 0;
    return cval;
}

template<typename To, typename Ti>
static To scalar(Ti real, Ti imag) {
    To cval;
    cval.s[0] = real;
    cval.s[1] = imag;
    return cval;
}

#ifdef AF_WITH_FAST_MATH
constexpr bool fast_math = true;
#else
constexpr bool fast_math = false;
#endif

template<typename T>
inline T maxval() {
    if constexpr (std::is_floating_point_v<T> && !fast_math) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}
template<typename T>
inline T minval() {
    if constexpr (std::is_floating_point_v<T> && !fast_math) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

static inline double real(cdouble in) { return in.s[0]; }
static inline float real(cfloat in) { return in.s[0]; }
static inline double imag(cdouble in) { return in.s[1]; }
static inline float imag(cfloat in) { return in.s[1]; }

cfloat operator+(cfloat lhs, cfloat rhs);
cfloat operator+(cfloat lhs);
cdouble operator+(cdouble lhs, cdouble rhs);
cdouble operator+(cdouble lhs);
cfloat operator*(cfloat lhs, cfloat rhs);
cdouble operator*(cdouble lhs, cdouble rhs);
common::half operator+(common::half lhs, common::half rhs) noexcept;
}  // namespace opencl
}  // namespace arrayfire

static inline bool operator==(arrayfire::opencl::cfloat lhs,
                              arrayfire::opencl::cfloat rhs) noexcept {
    return (lhs.s[0] == rhs.s[0]) && (lhs.s[1] == rhs.s[1]);
}
static inline bool operator!=(arrayfire::opencl::cfloat lhs,
                              arrayfire::opencl::cfloat rhs) noexcept {
    return !(lhs == rhs);
}
static inline bool operator==(arrayfire::opencl::cdouble lhs,
                              arrayfire::opencl::cdouble rhs) noexcept {
    return (lhs.s[0] == rhs.s[0]) && (lhs.s[1] == rhs.s[1]);
}
static inline bool operator!=(arrayfire::opencl::cdouble lhs,
                              arrayfire::opencl::cdouble rhs) noexcept {
    return !(lhs == rhs);
}

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic pop
#else
/* Other */
#endif
