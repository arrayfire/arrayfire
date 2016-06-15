/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <ops.hpp>

#undef _GLIBCXX_USE_INT128
#include <scan_by_key.hpp>
#include <complex>
#include <kernel/scan_first_by_key.hpp>
#include <kernel/scan_dim_by_key.hpp>

namespace cuda
{
    template<af_op_t op, typename Ti, typename Tk, typename To>
    Array<To> scan(const Array<Tk>& key, const Array<Ti>& in, const int dim, bool inclusive_scan)
    {
        Array<To> out = createEmptyArray<To>(in.dims());

        if (inclusive_scan) {
            switch (dim) {
            case 0: kernel::scan_first_by_key<Ti, Tk, To, op   , true>(out, in, key); break;
            case 1: kernel::scan_dim_by_key  <Ti, Tk, To, op, 1, true>(out, in, key); break;
            case 2: kernel::scan_dim_by_key  <Ti, Tk, To, op, 2, true>(out, in, key); break;
            case 3: kernel::scan_dim_by_key  <Ti, Tk, To, op, 3, true>(out, in, key); break;
            }
        } else {
            switch (dim) {
            case 0: kernel::scan_first_by_key<Ti, Tk, To, op   , false>(out, in, key); break;
            case 1: kernel::scan_dim_by_key  <Ti, Tk, To, op, 1, false>(out, in, key); break;
            case 2: kernel::scan_dim_by_key  <Ti, Tk, To, op, 2, false>(out, in, key); break;
            case 3: kernel::scan_dim_by_key  <Ti, Tk, To, op, 3, false>(out, in, key); break;
            }
        }

        return out;
    }

#define INSTANTIATE_SCAN_BY_KEY(ROp, Ti, Tk, To)\
    template Array<To> scan<ROp, Ti, Tk, To>(const Array<Tk>& key, const Array<Ti>& in, const int dim, bool inclusive_scan);

#define INSTANTIATE_SCAN_BY_KEY_ALL(ROp, Tk)            \
    INSTANTIATE_SCAN_BY_KEY(ROp, float  , Tk, float  )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, double , Tk, double )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, cfloat , Tk, cfloat )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, cdouble, Tk, cdouble)  \
    INSTANTIATE_SCAN_BY_KEY(ROp, int    , Tk, int    )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, uint   , Tk, uint   )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, intl   , Tk, intl   )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, uintl  , Tk, uintl  )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, char   , Tk, int    )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, char   , Tk, uint   )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, uchar  , Tk, uint   )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, short  , Tk, int    )  \
    INSTANTIATE_SCAN_BY_KEY(ROp, ushort , Tk, uint   )

#define INSTANTIATE_SCAN_OP(ROp)            \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, int  ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uint ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, intl ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uintl)

    INSTANTIATE_SCAN_OP(af_add_t)
    INSTANTIATE_SCAN_OP(af_mul_t)
    INSTANTIATE_SCAN_OP(af_min_t)
    INSTANTIATE_SCAN_OP(af_max_t)
}
