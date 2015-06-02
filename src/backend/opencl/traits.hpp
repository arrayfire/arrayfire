/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/traits.hpp>
#include <string>
#include <sstream>

namespace af
{

template<>
struct dtype_traits<cl_float2> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char* getName() { return "float2"; }
};

template<>
struct dtype_traits<cl_double2> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char* getName() { return "double2"; }
};

#if !defined(OS_WIN)        // Windows defines size_t as ulong
template<>
struct dtype_traits<size_t> {
    static const char* getName()
    {
        return (sizeof(size_t) == 8)  ? "ulong" : "uint";
    }
};
#endif

template<typename T> static bool iscplx() { return false; }
template<> STATIC_ bool iscplx<cl_float2>() { return true; }
template<> STATIC_ bool iscplx<cl_double2>() { return true; }

template<typename T>
STATIC_
std::string scalar_to_option(const T &val)
{
    return std::to_string(+val);
}

template<>
STATIC_
std::string scalar_to_option<cl_float2>(const cl_float2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}

template<>
STATIC_
std::string scalar_to_option<cl_double2>(const cl_double2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}
}

using af::dtype_traits;
