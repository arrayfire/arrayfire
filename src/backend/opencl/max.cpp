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
#define INSTANTIATE(Op, Ti, To)                                                 \
  template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim,     \
                                        bool change_nan, double nanval);        \
  template To reduce_all<Op, Ti, To>(const Array<Ti> &in,                       \
                                     bool change_nan, double nanval);
    //max
    INSTANTIATE(af_max_t, float  , float  )
    INSTANTIATE(af_max_t, double , double )
    INSTANTIATE(af_max_t, cfloat , cfloat )
    INSTANTIATE(af_max_t, cdouble, cdouble)
    INSTANTIATE(af_max_t, int    , int    )
    INSTANTIATE(af_max_t, uint   , uint   )
    INSTANTIATE(af_max_t, intl   , intl   )
    INSTANTIATE(af_max_t, uintl  , uintl  )
    INSTANTIATE(af_max_t, char   , char   )
    INSTANTIATE(af_max_t, uchar  , uchar  )
    INSTANTIATE(af_max_t, short  , short  )
    INSTANTIATE(af_max_t, ushort , ushort )

#undef INSTANTIATE
}
