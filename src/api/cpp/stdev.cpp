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

#define INSTANTIATE_VAR(T)                                \
    template<> AFAPI T stdev(const array& in)             \
    {                                                     \
        double ret_val;                                   \
        AF_THROW(af_stdev_all(&ret_val, NULL, in.get())); \
        return (T) ret_val;                               \
    }                                                     \

template<> AFAPI af_cfloat stdev(const array& in)
{
    double real, imag;
    AF_THROW(af_stdev_all(&real, &imag, in.get()));
    return std::complex<float>((float)real, (float)imag);
}

template<> AFAPI af_cdouble stdev(const array& in)
{
    double real, imag;
    AF_THROW(af_stdev_all(&real, &imag, in.get()));
    return std::complex<double>(real, imag);
}

INSTANTIATE_VAR(float);
INSTANTIATE_VAR(double);
INSTANTIATE_VAR(int);
INSTANTIATE_VAR(unsigned int);
INSTANTIATE_VAR(char);
INSTANTIATE_VAR(unsigned char);

#undef INSTANTIATE_VAR

array stdev(const array& in, dim_type dim)
{
    af_array temp = 0;
    AF_THROW(af_stdev(&temp, in.get(), getFNSD(in.dims())));
    return array(temp);
}

}
