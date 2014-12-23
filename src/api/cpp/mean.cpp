/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include <af/algorithm.h>
#include "error.hpp"

namespace af
{

array mean(const array &in, dim_type dim)
{
    af_array temp = 0;
    AF_THROW(af_mean(&temp, in.get(), dim));
    return array(temp);
}

#define INSTANTIATE_MEAN(T)                               \
    template<> AFAPI T mean(const array& in)              \
    {                                                     \
        double ret_val;                                   \
        AF_THROW(af_mean_all(&ret_val, NULL, in.get()));  \
        return (T)ret_val;                                \
    }                                                     \

template<> AFAPI af_cfloat mean(const array& in)
{
    double real, imag;
    AF_THROW(af_mean_all(&real, &imag, in.get()));
    return std::complex<float>((float)real, (float)imag);
}

template<> AFAPI af_cdouble mean(const array& in)
{
    double real, imag;
    AF_THROW(af_mean_all(&real, &imag, in.get()));
    return std::complex<double>(real, imag);
}


INSTANTIATE_MEAN(float);
INSTANTIATE_MEAN(double);
INSTANTIATE_MEAN(int);
INSTANTIATE_MEAN(unsigned int);
INSTANTIATE_MEAN(char);
INSTANTIATE_MEAN(unsigned char);

#undef INSTANTIATE_MEAN

}
