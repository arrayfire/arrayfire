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

namespace af
{

template<>
struct dtype_traits<cl_float2> {
    enum { af_type = c32 };
    static const char* getName() { return "float2"; }
};

template<>
struct dtype_traits<cl_double2> {
    enum { af_type = c64 };
    static const char* getName() { return "double2"; }
};

template<>
struct dtype_traits<size_t> {
    static const char* getName() { return "size_t"; };
};

template<>
struct dtype_traits<long long> {
    static const char* getName() { return "int"; };
};

template<typename T> static bool iscplx() { return false; }
template<> STATIC_ bool iscplx<cl_float2>() { return true; }
template<> STATIC_ bool iscplx<cl_double2>() { return true; }

}

using af::dtype_traits;
