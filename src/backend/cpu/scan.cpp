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
    Array<To> scan(const Array<Ti>& in, const int dim, bool inclusive_scan)
    {
        dim4 dims     = in.dims();
        Array<To> out = createEmptyArray<To>(dims);
        in.eval();

        if (inclusive_scan) {
            switch (in.ndims()) {
                case 1:
                    kernel::scan_dim<op, Ti, To, 1, true> func1;
                    getQueue().enqueue(func1, out, 0, in, 0, dim);
                    break;
                case 2:
                    kernel::scan_dim<op, Ti, To, 2, true> func2;
                    getQueue().enqueue(func2, out, 0, in, 0, dim);
                    break;
                case 3:
                    kernel::scan_dim<op, Ti, To, 3, true> func3;
                    getQueue().enqueue(func3, out, 0, in, 0, dim);
                    break;
                case 4:
                    kernel::scan_dim<op, Ti, To, 4, true> func4;
                    getQueue().enqueue(func4, out, 0, in, 0, dim);
                    break;
            }
        } else {
            switch (in.ndims()) {
                case 1:
                    kernel::scan_dim<op, Ti, To, 1, false> func1;
                    getQueue().enqueue(func1, out, 0, in, 0, dim);
                    break;
                case 2:
                    kernel::scan_dim<op, Ti, To, 2, false> func2;
                    getQueue().enqueue(func2, out, 0, in, 0, dim);
                    break;
                case 3:
                    kernel::scan_dim<op, Ti, To, 3, false> func3;
                    getQueue().enqueue(func3, out, 0, in, 0, dim);
                    break;
                case 4:
                    kernel::scan_dim<op, Ti, To, 4, false> func4;
                    getQueue().enqueue(func4, out, 0, in, 0, dim);
                    break;
            }
        }

        return out;
    }

#define INSTANTIATE(ROp, Ti, To)\
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim, bool inclusive_scan);

#define INSTANTIATE_SCAN(ROp)           \
    INSTANTIATE(ROp, float  , float  )  \
    INSTANTIATE(ROp, double , double )  \
    INSTANTIATE(ROp, cfloat , cfloat )  \
    INSTANTIATE(ROp, cdouble, cdouble)  \
    INSTANTIATE(ROp, int    , int    )  \
    INSTANTIATE(ROp, uint   , uint   )  \
    INSTANTIATE(ROp, intl   , intl   )  \
    INSTANTIATE(ROp, uintl  , uintl  )  \
    INSTANTIATE(ROp, char   , int    )  \
    INSTANTIATE(ROp, char   , uint   )  \
    INSTANTIATE(ROp, uchar  , uint   )  \
    INSTANTIATE(ROp, short  , int    )  \
    INSTANTIATE(ROp, ushort , uint   )

    //accum
    INSTANTIATE(af_notzero_t, char  , uint)
    INSTANTIATE_SCAN(af_add_t)
    INSTANTIATE_SCAN(af_sub_t)
    INSTANTIATE_SCAN(af_mul_t)
    INSTANTIATE_SCAN(af_div_t)
    INSTANTIATE_SCAN(af_min_t)
    INSTANTIATE_SCAN(af_max_t)
}
