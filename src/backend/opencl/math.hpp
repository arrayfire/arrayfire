/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <complex>
#include <limits>
#include <algorithm>
#include "backend.hpp"
#include "types.hpp"

namespace opencl
{

#undef min
#undef max

    template<typename T> static inline T abs(T val)  { return std::abs(val); }
    template<typename T> static inline T min(T lhs, T rhs) { return std::min(lhs, rhs); }
    template<typename T> static inline T max(T lhs, T rhs) { return std::max(lhs, rhs); }

    static inline float  abs(cfloat  cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }
    static inline double abs(cdouble cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }

    template<> STATIC_
    cfloat max<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

	template<> STATIC_
    cdouble max<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

	template<> STATIC_
    cfloat min<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

	template<> STATIC_
    cdouble min<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<typename T>
    static T scalar(double val)
    {
        return (T)(val);
    }

	template<> STATIC_
    cfloat  scalar<cfloat >(double val)
    {
        cfloat  cval;
        cval.s[0]= (float)val;
        cval.s[1] = 0;
        return cval;
    }

	template<> STATIC_
    cdouble scalar<cdouble >(double val)
    {
        cdouble cval;
        cval.s[0]= val;
        cval.s[1] = 0;
        return cval;
    }

    template <typename T> T limit_max() { return std::numeric_limits<T>::max(); }
    template <typename T> T limit_min() { return std::numeric_limits<T>::min(); }

    static inline double real(cdouble in)
    {
        return in.s[0];
    }
    static inline float real(cfloat in)
    {
        return in.s[0];
    }
    static inline double imag(cdouble in)
    {
        return in.s[1];
    }
    static inline float imag(cfloat in)
    {
        return in.s[1];
    }
}
