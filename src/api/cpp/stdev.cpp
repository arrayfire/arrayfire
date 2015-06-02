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
#include <af/array.h>
#include "error.hpp"
#include "common.hpp"

namespace af
{

#define INSTANTIATE_STDEV(T)                              \
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
    return af_cfloat((float)real, (float)imag);
}

template<> AFAPI af_cdouble stdev(const array& in)
{
    double real, imag;
    AF_THROW(af_stdev_all(&real, &imag, in.get()));
    return af_cdouble(real, imag);
}

INSTANTIATE_STDEV(float);
INSTANTIATE_STDEV(double);
INSTANTIATE_STDEV(int);
INSTANTIATE_STDEV(unsigned int);
INSTANTIATE_STDEV(char);
INSTANTIATE_STDEV(unsigned char);

#undef INSTANTIATE_STDEV

array stdev(const array& in, const dim_t dim)
{
    af_array temp = 0;
    AF_THROW(af_stdev(&temp, in.get(), getFNSD(dim, in.dims())));
    return array(temp);
}

}
