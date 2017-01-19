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
#include <sstream>
#include <string>
#include <cmath>
#include <type_util.hpp>

namespace opencl
{
typedef cl_float2   cfloat;
typedef cl_double2 cdouble;
typedef cl_uchar     uchar;
typedef cl_uint       uint;
typedef cl_ushort   ushort;

template<typename T> struct is_complex          { static const bool value = false;  };
template<> struct           is_complex<cfloat>  { static const bool value = true;   };
template<> struct           is_complex<cdouble> { static const bool value = true;   };

template<typename T > const char *shortname(bool caps=false);

template<typename T>
struct ToNumStr
{
    inline std::string operator()(T val)
    {
        ToNum<T> toNum;
        return std::to_string(toNum(val));
    }
};

template<>
struct ToNumStr<float>
{
    inline std::string operator()(float val)
    {
        static const std::string PINF = "+INFINITY";
        static const std::string NINF = "-INFINITY";
        if (std::isinf(val)) {
            return val < 0 ? NINF : PINF;
        }
        return std::to_string(val);
    }
};

template<>
struct ToNumStr<double>
{
    inline std::string operator()(double val)
    {
        static const std::string PINF = "+INFINITY";
        static const std::string NINF = "-INFINITY";
        if (std::isinf(val)) {
            return val < 0 ? NINF : PINF;
        }
        return std::to_string(val);
    }
};

template<>
struct ToNumStr<cfloat>
{
    inline std::string operator()(cfloat val)
    {
        ToNumStr<float> realStr;
        static const std::string INF = "INFINITY";
        std::stringstream s;
        s << "{";
        s << realStr(val.s[0]);
        s << ",";
        s << realStr(val.s[1]);
        s << "}";
        return s.str();
    }
};

template<>
struct ToNumStr<cdouble>
{
    inline std::string operator()(cdouble val)
    {
        ToNumStr<double> realStr;
        static const std::string INF = "INFINITY";
        std::stringstream s;
        s << "{";
        s << realStr(val.s[0]);
        s << ",";
        s << realStr(val.s[1]);
        s << "}";
        return s.str();
    }
};
}
