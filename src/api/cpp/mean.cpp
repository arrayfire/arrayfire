/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/statistics.h>
#include <af/algorithm.h>
#include "error.hpp"

static inline dim_type getFNSD(af::dim4 dims)
{
    dim_type fNSD = 0;
    for (dim_type i=0; i<4; ++i) {
        if (dims[i]>1) {
            fNSD = i;
            break;
        }
    }
    return fNSD;
}

namespace af
{

array mean(const array &in, dim_type dim)
{
    af_array temp = 0;
    AF_THROW(af_mean(&temp, in.get(), getFNSD(in.dims())));
    return array(temp);
}

array mean(const array &in, const array &weights, dim_type dim)
{
    af_array temp = 0;
    AF_THROW(af_mean_weighted(&temp, in.get(), weights.get(), getFNSD(in.dims())));
    return array(temp);
}

#define INSTANTIATE_MEAN(T)                                     \
    template<> AFAPI T mean(const array& in)                    \
    {                                                           \
        double ret_val;                                         \
        AF_THROW(af_mean_all(&ret_val, NULL, in.get()));        \
        return (T)ret_val;                                      \
    }                                                           \
    template<> AFAPI T mean(const array& in, const array& wts)  \
    {                                                           \
        double ret_val;                                         \
        AF_THROW(af_mean_all_weighted(&ret_val, NULL,           \
                    in.get(), wts.get()));                      \
        return (T)ret_val;                                      \
    }                                                           \

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

template<> AFAPI af_cfloat mean(const array& in, const array& weights)
{
    double real, imag;
    AF_THROW(af_mean_all_weighted(&real, &imag, in.get(), weights.get()));
    return std::complex<float>((float)real, (float)imag);
}

template<> AFAPI af_cdouble mean(const array& in, const array& weights)
{
    double real, imag;
    AF_THROW(af_mean_all_weighted(&real, &imag, in.get(), weights.get()));
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
