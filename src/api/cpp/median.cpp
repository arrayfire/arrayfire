/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include <af/array.h>
#include "error.hpp"
#include "common.hpp"

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

AFAPI array median(const array& in, const dim_t dim)
{
    af_array temp = 0;
    AF_THROW(af_median(&temp, in.get(), getFNSD(dim, in.dims())));
    return array(temp);
}

}
