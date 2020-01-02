/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#pragma GCC diagnostic pop

#include <common/kernel_type.hpp>
#include <common/traits.hpp>

#include <string>

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
inline const char *shortname(bool caps) {
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
}  // namespace

template<typename T>
const char *getFullName() {
    return af::dtype_traits<T>::getName();
}

}  // namespace opencl
