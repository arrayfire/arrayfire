/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <scan.hpp>
#include <complex>
#include <kernel/scan_first.hpp>
#include <kernel/scan_dim.hpp>

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To> scan(const Array<Ti>& in, const int dim, bool inclusive_scan)
    {
        Array<To> out = createEmptyArray<To>(in.dims());

        if (inclusive_scan) {
            if(dim == 0) {
               kernel::scan_first<Ti, To, op, true>(out, in);
            } else {
                kernel::scan_dim <Ti, To, op, true>(out, in, dim);
            }
        } else {
            if(dim == 0) {
               kernel::scan_first<Ti, To, op, false>(out, in);
            } else {
                kernel::scan_dim <Ti, To, op, false>(out, in, dim);
            }
        }

        return out;
    }

#define INSTANTIATE_SCAN(ROp, Ti, To)\
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim, bool inclusive_scan);

#define INSTANTIATE_SCAN_ALL(ROp)                       \
    INSTANTIATE_SCAN(ROp, float  , float  )             \
    INSTANTIATE_SCAN(ROp, double , double )             \
    INSTANTIATE_SCAN(ROp, cfloat , cfloat )             \
    INSTANTIATE_SCAN(ROp, cdouble, cdouble)             \
    INSTANTIATE_SCAN(ROp, int    , int    )             \
    INSTANTIATE_SCAN(ROp, uint   , uint   )             \
    INSTANTIATE_SCAN(ROp, intl   , intl   )             \
    INSTANTIATE_SCAN(ROp, uintl  , uintl  )             \
    INSTANTIATE_SCAN(ROp, char   , int    )             \
    INSTANTIATE_SCAN(ROp, char   , uint   )             \
    INSTANTIATE_SCAN(ROp, uchar  , uint   )             \
    INSTANTIATE_SCAN(ROp, short  , int    )             \
    INSTANTIATE_SCAN(ROp, ushort , uint   )

}
