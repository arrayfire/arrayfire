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

#define INSTANTIATE_MEDIAN(T)                               \
    template<> AFAPI T median(const array& in)              \
    {                                                       \
        double ret_val;                                     \
        AF_THROW(af_median_all(&ret_val, NULL, in.get()));  \
        return (T)ret_val;                                  \
    }                                                       \

INSTANTIATE_MEDIAN(float);
INSTANTIATE_MEDIAN(double);
INSTANTIATE_MEDIAN(int);
INSTANTIATE_MEDIAN(unsigned int);
INSTANTIATE_MEDIAN(char);
INSTANTIATE_MEDIAN(unsigned char);

#undef INSTANTIATE_MEDIAN

AFAPI array median(const array& in, dim_type dim)
{
    af_array temp = 0;
    AF_THROW(af_median(&temp, in.get(), getFNSD(in.dims())));
    return array(temp);
}

}

