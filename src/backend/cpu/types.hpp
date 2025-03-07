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
#include <complex>

namespace arrayfire {
namespace cpu {

namespace {
template<typename T>
const char *shortname(bool caps = false) {
    return caps ? "?" : "?";
}

template<typename T>
const char *getFullName() {
    return "N/A";
}

}  // namespace

using cdouble = std::complex<double>;
using cfloat  = std::complex<float>;
using intl    = long long;
using uint    = unsigned int;
using uchar   = unsigned char;
using uintl   = unsigned long long;
using ushort  = unsigned short;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

}  // namespace cpu

namespace common {
template<typename T>
struct kernel_type;

class half;

template<>
struct kernel_type<arrayfire::common::half> {
    using data = arrayfire::common::half;

    // These are the types within a kernel
    using native = float;

    using compute = float;
};
}  // namespace common

}  // namespace arrayfire
