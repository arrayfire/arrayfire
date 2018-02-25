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

#define INSTANTIATE(Op, Ti, To)                                             \
  template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim, \
                                        bool change_nan, double nanval);    \
  template To reduce_all<Op, Ti, To>(const Array<Ti> &in,                   \
                                     bool change_nan, double nanval);
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
    INSTANTIATE(af_notzero_t, short  , uint)
    INSTANTIATE(af_notzero_t, ushort , uint)

#undef INSTANTIATE
}
