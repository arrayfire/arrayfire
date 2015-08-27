/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "reduce_impl.hpp"

namespace opencl
{
    // count
    INSTANTIATE(af_notzero_t, float  , uint)
    INSTANTIATE(af_notzero_t, double , uint)
    INSTANTIATE(af_notzero_t, cfloat , uint)
    INSTANTIATE(af_notzero_t, cdouble, uint)
    INSTANTIATE(af_notzero_t, int    , uint)
    INSTANTIATE(af_notzero_t, uint   , uint)
    INSTANTIATE(af_notzero_t, intl   , uint)
    INSTANTIATE(af_notzero_t, uintl  , uint)
    INSTANTIATE(af_notzero_t, char   , uint)
    INSTANTIATE(af_notzero_t, uchar  , uint)
}
