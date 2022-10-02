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
#include <common/traits.hpp>
#include <types.hpp>

#include <sstream>
#include <string>

namespace af {

template<>
struct dtype_traits<opencl::cfloat> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char *getName() { return "float2"; }
};

template<>
struct dtype_traits<opencl::cdouble> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char *getName() { return "double2"; }
};

template<typename T>
static bool iscplx() {
    return false;
}
template<>
inline bool iscplx<opencl::cfloat>() {
    return true;
}
template<>
inline bool iscplx<opencl::cdouble>() {
    return true;
}

template<typename T>
inline std::string scalar_to_option(const T &val) {
    using namespace common;
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
