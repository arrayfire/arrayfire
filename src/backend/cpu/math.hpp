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
#include <numeric>
#include "types.hpp"
#include <af/defines.h>

namespace cpu
{
    template<typename T> static inline T abs(T val) { return std::abs(val); }
    uint abs(uint val);
    uchar abs(uchar val);
    uintl abs(uintl val);
#if !(defined(OS_WIN) || (defined(ARCH_32) && defined(OS_LNX)))  // Not(Windows or Tegra)
    size_t abs(size_t val);
#endif

    template<typename T> static inline T min(T lhs, T rhs) { return std::min(lhs, rhs); }
    cfloat min(cfloat lhs, cfloat rhs);
    cdouble min(cdouble lhs, cdouble rhs);

    template<typename T> static inline T max(T lhs, T rhs) { return std::max(lhs, rhs); }
    cfloat max(cfloat lhs, cfloat rhs);
    cdouble max(cdouble lhs, cdouble rhs);

    template<typename T> static inline T division(T lhs, double rhs) { return lhs / rhs; }

    template<> STATIC_ cfloat division<cfloat>(cfloat lhs, double rhs)
    {
        cfloat retVal(real(lhs) / rhs, imag(lhs) / rhs);
        return retVal;
    }

    template<> STATIC_ cdouble division<cdouble>(cdouble lhs, double rhs)
    {
        cdouble retVal(real(lhs) / rhs, imag(lhs) / rhs);
        return retVal;
    }

    template <typename T> static inline T limit_max()
    { return std::numeric_limits<T>::max(); }

    template <typename T> static inline T limit_min()
    { return std::numeric_limits<T>::min(); }

    template<typename T>
    static T scalar(double val)
    {
        return (T)(val);
    }

    template<typename To, typename Ti>
    static To scalar(Ti real, Ti imag)
    {
        To  cval = {real, imag};
        return cval;
    }

    cfloat  scalar(float val);

    cdouble scalar(double val);
}
