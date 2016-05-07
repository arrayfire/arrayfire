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
            switch (dim) {
            case 0: kernel::scan_first<Ti, To, op   , true>(out, in); break;
            case 1: kernel::scan_dim  <Ti, To, op, 1, true>(out, in); break;
            case 2: kernel::scan_dim  <Ti, To, op, 2, true>(out, in); break;
            case 3: kernel::scan_dim  <Ti, To, op, 3, true>(out, in); break;
            }
        } else {
            switch (dim) {
            case 0: kernel::scan_first<Ti, To, op   , false>(out, in); break;
            case 1: kernel::scan_dim  <Ti, To, op, 1, false>(out, in); break;
            case 2: kernel::scan_dim  <Ti, To, op, 2, false>(out, in); break;
            case 3: kernel::scan_dim  <Ti, To, op, 3, false>(out, in); break;
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
