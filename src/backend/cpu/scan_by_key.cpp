/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/scan_by_key.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <scan_by_key.hpp>
#include <af/dim4.hpp>
#include <complex>

using af::dim4;

namespace arrayfire {
namespace cpu {
template<af_op_t op, typename Ti, typename Tk, typename To>
Array<To> scan(const Array<Tk>& key, const Array<Ti>& in, const int dim,
               bool inclusive_scan) {
    const dim4& dims = in.dims();
    Array<To> out    = createEmptyArray<To>(dims);
    kernel::scan_dim_by_key<op, Ti, Tk, To, 1> func1(inclusive_scan);
    kernel::scan_dim_by_key<op, Ti, Tk, To, 2> func2(inclusive_scan);
    kernel::scan_dim_by_key<op, Ti, Tk, To, 3> func3(inclusive_scan);
    kernel::scan_dim_by_key<op, Ti, Tk, To, 4> func4(inclusive_scan);

    switch (in.ndims()) {
        case 1: getQueue().enqueue(func1, out, 0, key, 0, in, 0, dim); break;
        case 2: getQueue().enqueue(func2, out, 0, key, 0, in, 0, dim); break;
        case 3: getQueue().enqueue(func3, out, 0, key, 0, in, 0, dim); break;
        case 4: getQueue().enqueue(func4, out, 0, key, 0, in, 0, dim); break;
    }

    return out;
}

#define INSTANTIATE_SCAN_BY_KEY(ROp, Ti, Tk, To)                  \
    template Array<To> scan<ROp, Ti, Tk, To>(                     \
        const Array<Tk>& key, const Array<Ti>& in, const int dim, \
        bool inclusive_scan);

#define INSTANTIATE_SCAN_BY_KEY_ALL(ROp, Tk)           \
    INSTANTIATE_SCAN_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_BY_KEY_ALL_OP(ROp) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, int)   \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uint)  \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, intl)  \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uintl)

INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_add_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_mul_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_min_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_max_t)
}  // namespace cpu
}  // namespace arrayfire
