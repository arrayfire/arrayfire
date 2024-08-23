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
#include <types.hpp>
#include <af/defines.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <numeric>

namespace arrayfire {
namespace cpu {
template<typename T>
static inline T abs(T val) {
    return std::abs(val);
}
uint abs(uint val);
uchar abs(uchar val);
uintl abs(uintl val);

template<typename T>
static inline T min(T lhs, T rhs) {
    return std::min(lhs, rhs);
}
cfloat min(cfloat lhs, cfloat rhs);
cdouble min(cdouble lhs, cdouble rhs);

template<typename T>
static inline T max(T lhs, T rhs) {
    return std::max(lhs, rhs);
}
cfloat max(cfloat lhs, cfloat rhs);
cdouble max(cdouble lhs, cdouble rhs);

template<typename T>
static inline auto is_nan(const T &val) -> bool {
    return false;
}

template<>
inline auto is_nan<float>(const float &val) -> bool {
    return std::isnan(val);
}

template<>
inline auto is_nan<double>(const double &val) -> bool {
    return std::isnan(val);
}

template<>
inline auto is_nan<common::half>(const common::half &val) -> bool {
    return isnan(val);
}

template<>
inline auto is_nan<cfloat>(const cfloat &in) -> bool {
    return std::isnan(real(in)) || std::isnan(imag(in));
}

template<>
inline auto is_nan<cdouble>(const cdouble &in) -> bool {
    return std::isnan(real(in)) || std::isnan(imag(in));
}

template<typename T>
static inline T division(T lhs, double rhs) {
    return lhs / rhs;
}

template<>
inline cfloat division<cfloat>(cfloat lhs, double rhs) {
    cfloat retVal(real(lhs) / static_cast<float>(rhs),
                  imag(lhs) / static_cast<float>(rhs));
    return retVal;
}

template<>
inline cdouble division<cdouble>(cdouble lhs, double rhs) {
    cdouble retVal(real(lhs) / rhs, imag(lhs) / rhs);
    return retVal;
}

template<typename T>
inline T maxval() {
    return std::numeric_limits<T>::max();
}
template<typename T>
inline T minval() {
    return std::numeric_limits<T>::lowest();
}
template<>
inline float maxval() {
    return std::numeric_limits<float>::infinity();
}
template<>
inline double maxval() {
    return std::numeric_limits<double>::infinity();
}
template<>
inline arrayfire::common::half maxval() {
    return std::numeric_limits<arrayfire::common::half>::infinity();
}
template<>
inline float minval() {
    return -std::numeric_limits<float>::infinity();
}
template<>
inline double minval() {
    return -std::numeric_limits<double>::infinity();
}
template<>
inline arrayfire::common::half minval() {
    return -std::numeric_limits<arrayfire::common::half>::infinity();
}

template<typename T>
static T scalar(double val) {
    return T(val);
}

template<typename To, typename Ti>
static To scalar(Ti real, Ti imag) {
    To cval = {real, imag};
    return cval;
}

cfloat scalar(float val);

cdouble scalar(double val);

inline double real(cdouble in) noexcept { return std::real(in); }
inline float real(cfloat in) noexcept { return std::real(in); }
inline double imag(cdouble in) noexcept { return std::imag(in); }
inline float imag(cfloat in) noexcept { return std::imag(in); }

}  // namespace cpu
}  // namespace arrayfire
