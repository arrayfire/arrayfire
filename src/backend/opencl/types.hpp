/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cl2hpp.hpp>
#include <common/kernel_type.hpp>
#include <common/traits.hpp>
#include <af/compilers.h>

#include <algorithm>
#include <array>
#include <string>

namespace arrayfire {
namespace common {
/// This is a CPU based half which need to be converted into floats before they
/// are used
template<>
struct kernel_type<common::half> {
    using data = common::half;

    // These are the types within a kernel
    using native = float;

    using compute = float;
};
}  // namespace common
}  // namespace arrayfire

namespace arrayfire {
namespace opencl {
using cdouble = cl_double2;
using cfloat  = cl_float2;
using intl    = long long;
using uchar   = cl_uchar;
using uint    = cl_uint;
using uintl   = unsigned long long;
using ushort  = cl_ushort;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

template<typename T>
struct ToNumStr {
    std::string operator()(T val);
    template<typename CONVERSION_TYPE>
    std::string operator()(CONVERSION_TYPE val);
};

namespace {
template<typename T>
inline const char *shortname(bool caps = false) {
    return caps ? "X" : "x";
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
    return caps ? "L" : "l";
}
template<>
inline const char *shortname<uintl>(bool caps) {
    return caps ? "K" : "k";
}
template<>
inline const char *shortname<short>(bool caps) {
    return caps ? "P" : "p";
}
template<>
inline const char *shortname<ushort>(bool caps) {
    return caps ? "Q" : "q";
}

template<typename T>
inline const char *getFullName() {
    return af::dtype_traits<T>::getName();
}

template<>
inline const char *getFullName<cfloat>() {
    return "float2";
}

template<>
inline const char *getFullName<cdouble>() {
    return "double2";
}
}  // namespace

template<typename... ARGS>
AF_CONSTEXPR const char *getTypeBuildDefinition() {
    using arrayfire::common::half;
    using std::any_of;
    using std::array;
    using std::begin;
    using std::end;
    using std::is_same;
    array<bool, sizeof...(ARGS)> is_half    = {is_same<ARGS, half>::value...};
    array<bool, sizeof...(ARGS)> is_double  = {is_same<ARGS, double>::value...};
    array<bool, sizeof...(ARGS)> is_cdouble = {
        is_same<ARGS, cdouble>::value...};

    bool half_def =
        any_of(begin(is_half), end(is_half), [](bool val) { return val; });
    bool double_def =
        any_of(begin(is_double), end(is_double), [](bool val) { return val; });
    bool cdouble_def = any_of(begin(is_cdouble), end(is_cdouble),
                              [](bool val) { return val; });

    if (half_def && (double_def || cdouble_def)) {
        return " -D USE_HALF -D USE_DOUBLE";
    } else if (half_def) {
        return " -D USE_HALF";
    } else if (double_def || cdouble_def) {
        return " -D USE_DOUBLE";
    } else {
        return "";
    }
}

}  // namespace opencl
}  // namespace arrayfire
