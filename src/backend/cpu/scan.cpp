/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <scan.hpp>
#include <ops.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/scan.hpp>

using af::dim4;

namespace cpu
{

template<af_op_t op, typename Ti, typename To>
Array<To> scan(const Array<Ti>& in, const int dim)
{
    dim4 dims     = in.dims();
    Array<To> out = createEmptyArray<To>(dims);
    in.eval();

    switch (in.ndims()) {
        case 1:
            kernel::scan_dim<op, Ti, To, 1> func1;
            getQueue().enqueue(func1, out, 0, in, 0, dim);
            break;
        case 2:
            kernel::scan_dim<op, Ti, To, 2> func2;
            getQueue().enqueue(func2, out, 0, in, 0, dim);
            break;
        case 3:
            kernel::scan_dim<op, Ti, To, 3> func3;
            getQueue().enqueue(func3, out, 0, in, 0, dim);
            break;
        case 4:
            kernel::scan_dim<op, Ti, To, 4> func4;
            getQueue().enqueue(func4, out, 0, in, 0, dim);
            break;
    }

    return out;
}

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim); \

//accum
INSTANTIATE(af_add_t, float  , float  )
INSTANTIATE(af_add_t, double , double )
INSTANTIATE(af_add_t, cfloat , cfloat )
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int    , int    )
INSTANTIATE(af_add_t, uint   , uint   )
INSTANTIATE(af_add_t, intl   , intl   )
INSTANTIATE(af_add_t, uintl  , uintl  )
INSTANTIATE(af_add_t, char   , int    )
INSTANTIATE(af_add_t, uchar  , uint   )
INSTANTIATE(af_add_t, short  , int    )
INSTANTIATE(af_add_t, ushort , uint   )
INSTANTIATE(af_notzero_t, char  , uint)

}
