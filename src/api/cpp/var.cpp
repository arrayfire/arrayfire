/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include "error.hpp"

namespace af
{

array var(const array& in, bool isbiased, int dim)
{
    af_array temp = 0;
    return array(temp);
}

array var(const array& in, const array weights, int dim)
{
    af_array temp = 0;
    return array(temp);
}

#define INSTANTIATE_VAR(T)                                          \
    template<> AFAPI T var(const array& in, bool isbiased)          \
    {                                                               \
        double ret_val;                                             \
        af_var_all(&ret_val, NULL, in.get(), isbiased);             \
        return (T) ret_val;                                         \
    }                                                               \
                                                                    \
    template<> AFAPI T var(const array& in, const array weights)    \
    {                                                               \
        double ret_val;                                             \
        return (T) ret_val;                                         \
    }                                                               \

template<> AFAPI af_cfloat var(const array& in, bool isbiased)
{
    double real, imag;
    AF_THROW(af_var_all(&real, &imag, in.get(), isbiased));
    return std::complex<float>((float)real, (float)imag);
}

template<> AFAPI af_cdouble var(const array& in, bool isbiased)
{
    double real, imag;
    AF_THROW(af_var_all(&real, &imag, in.get(), isbiased));
    return std::complex<double>(real, imag);
}

template<> AFAPI af_cfloat var(const array& in, const array weights)
{
    double real, imag;
    return std::complex<float>((float)real, (float)imag);
}

template<> AFAPI af_cdouble var(const array& in, const array weights)
{
    double real, imag;
    return std::complex<double>(real, imag);
}

INSTANTIATE_VAR(float);
INSTANTIATE_VAR(double);
INSTANTIATE_VAR(int);
INSTANTIATE_VAR(unsigned int);
INSTANTIATE_VAR(char);
INSTANTIATE_VAR(unsigned char);

#undef INSTANTIATE_VAR

}
