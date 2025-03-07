/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <common/traits.hpp>
#include <types.hpp>

#include <sstream>
#include <string>

namespace af {

template<typename T>
static bool iscplx() {
    return false;
}
template<>
inline bool iscplx<arrayfire::oneapi::cfloat>() {
    return true;
}
template<>
inline bool iscplx<arrayfire::oneapi::cdouble>() {
    return true;
}

template<typename T>
inline std::string scalar_to_option(const T &val) {
    using namespace arrayfire::common;
    using namespace std;
    return to_string(+val);
}

template<>
inline std::string scalar_to_option<cl_float2>(const cl_float2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}

template<>
inline std::string scalar_to_option<cl_double2>(const cl_double2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}
}  // namespace af

using af::dtype_traits;
