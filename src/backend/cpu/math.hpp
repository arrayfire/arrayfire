/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <limits>
#include <algorithm>
#include "types.hpp"

namespace cpu
{
    template<typename T> static inline T abs(T val) { return std::abs(val); }
    template<typename T> static inline T min(T lhs, T rhs) { return std::min(lhs, rhs); }
    template<typename T> static inline T max(T lhs, T rhs) { return std::max(lhs, rhs); }

    static inline float  abs(cfloat  cval) { return std::abs(cval); }
    static inline double abs(cdouble cval) { return std::abs(cval); }

    static
    cfloat max(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

	static
    cdouble max(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

	static
    cfloat min(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

	static
    cdouble min(cdouble lhs, cdouble rhs)
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
        cfloat  cval = {(float)val, 0};
        return cval;
    }

	template<> STATIC_
    cdouble scalar<cdouble >(double val)
    {
        cdouble  cval = {val, 0};
        return cval;
    }

    template <typename T> T limit_max() { return std::numeric_limits<T>::max(); }
    template <typename T> T limit_min() { return std::numeric_limits<T>::min(); }
}
