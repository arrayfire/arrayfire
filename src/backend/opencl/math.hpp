#pragma once
#include <complex>
#include <limits>
#include "backend.hpp"
#include "types.hpp"

namespace opencl
{

    template<typename T> static inline T abs(T val)  { return std::abs(val); }
    template<typename T> static inline T min(T lhs, T rhs) { return std::min(lhs, rhs); }
    template<typename T> static inline T max(T lhs, T rhs) { return std::max(lhs, rhs); }

    static inline float  abs(cfloat  cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }
    static inline double abs(cdouble cval) { return std::sqrt(cval.s[0]*cval.s[0] + cval.s[1]*cval.s[1]); }

    template<>
    cfloat max<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<>
    cdouble max<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<>
    cfloat min<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<>
    cdouble min<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<typename T>
    static T scalar(double val)
    {
        return (T)(val);
    }

    template<>
    cfloat  scalar<cfloat >(double val)
    {
        cfloat  cval = {(float)val, 0};
        return cval;
    }

    template<>
    cdouble scalar<cdouble >(double val)
    {
        cdouble  cval = {val, 0};
        return cval;
    }

    template <typename T> T limit_max() { return std::numeric_limits<T>::max(); }
    template <typename T> T limit_min() { return std::numeric_limits<T>::min(); }
}
