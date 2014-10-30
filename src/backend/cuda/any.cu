/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "reduce_impl.hpp"

namespace cuda
{
    //anytrue
    INSTANTIATE(af_or_t, float  , uchar)
    INSTANTIATE(af_or_t, double , uchar)
    INSTANTIATE(af_or_t, cfloat , uchar)
    INSTANTIATE(af_or_t, cdouble, uchar)
    INSTANTIATE(af_or_t, int    , uchar)
    INSTANTIATE(af_or_t, uint   , uchar)
    INSTANTIATE(af_or_t, char   , uchar)
    INSTANTIATE(af_or_t, uchar  , uchar)
}
