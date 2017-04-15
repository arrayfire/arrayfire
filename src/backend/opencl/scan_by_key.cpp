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
#include <scan.hpp>
#include <complex>
#include <err_opencl.hpp>

#include <kernel/scan_first_by_key.hpp>
#include <kernel/scan_dim_by_key.hpp>

namespace opencl
{
    template<af_op_t op, typename Ti, typename Tk, typename To>
    Array<To> scan(const Array<Tk>& key, const Array<Ti>& in, const int dim, bool inclusive_scan)
    {
        Array<To> out = createEmptyArray<To>(in.dims());

        Param Out = out;
        Param Key = key;
        Param In  =   in;

        if (inclusive_scan) {
            if (dim == 0)
                kernel::scan_first<Ti, Tk, To, op, true>(Out, In, Key);
            else
                kernel::scan_dim  <Ti, Tk, To, op, true>(Out, In, Key, dim);
        } else {
            if (dim == 0)
                kernel::scan_first<Ti, Tk, To, op, false>(Out, In, Key);
            else
                kernel::scan_dim  <Ti, Tk, To, op, false>(Out, In, Key, dim);
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

#define INSTANTIATE_SCAN_BY_KEY_OP(ROp)     \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, int  ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uint ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, intl ) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uintl)

    INSTANTIATE_SCAN_BY_KEY_OP(af_add_t)
    INSTANTIATE_SCAN_BY_KEY_OP(af_mul_t)
    INSTANTIATE_SCAN_BY_KEY_OP(af_min_t)
    INSTANTIATE_SCAN_BY_KEY_OP(af_max_t)
}
