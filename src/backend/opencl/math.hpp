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
#include <af/defines.h>

#include <backend.hpp>
#include <types.hpp>

#include <algorithm>
#include <complex>
#include <limits>

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#else
/* Other */
#endif

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
STATIC_ cfloat max<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
STATIC_ cdouble max<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
STATIC_ cfloat min<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<>
STATIC_ cdouble min<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<typename T>
static T scalar(double val) {
    return (T)(val);
}

template<>
STATIC_ cfloat scalar<cfloat>(double val) {
    cfloat cval;
    cval.s[0] = (float)val;
    cval.s[1] = 0;
    return cval;
}

template<>
STATIC_ cdouble scalar<cdouble>(double val) {
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

template<typename T>
STATIC_ T maxval() {
    return std::numeric_limits<T>::max();
}
template<typename T>
STATIC_ T minval() {
    return std::numeric_limits<T>::min();
}
template<>
STATIC_ float maxval() {
    return std::numeric_limits<float>::infinity();
}
template<>
STATIC_ double maxval() {
    return std::numeric_limits<double>::infinity();
}
template<>
STATIC_ float minval() {
    return -std::numeric_limits<float>::infinity();
}
template<>
STATIC_ double minval() {
    return -std::numeric_limits<double>::infinity();
}

static inline double real(cdouble in) { return in.s[0]; }
static inline float real(cfloat in) { return in.s[0]; }
static inline double imag(cdouble in) { return in.s[1]; }
static inline float imag(cfloat in) { return in.s[1]; }

bool operator==(cfloat a, cfloat b);
bool operator!=(cfloat a, cfloat b);
bool operator==(cdouble a, cdouble b);
bool operator!=(cdouble a, cdouble b);
cfloat operator+(cfloat a, cfloat b);
cfloat operator+(cfloat a);
cdouble operator+(cdouble a, cdouble b);
cdouble operator+(cdouble a);
cfloat operator*(cfloat lhs, cfloat rhs);
cdouble operator*(cdouble lhs, cdouble rhs);
common::half operator+(common::half a, common::half b) noexcept;
}  // namespace opencl

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic pop
#else
/* Other */
#endif
