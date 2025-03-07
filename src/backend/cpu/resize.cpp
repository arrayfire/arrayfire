/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/resize.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <resize.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                const af_interp_type method) {
    af::dim4 idims = in.dims();
    af::dim4 odims(odim0, odim1, idims[2], idims[3]);
    // Create output placeholder
    Array<T> out = createValueArray(odims, static_cast<T>(0));

    switch (method) {
        case AF_INTERP_NEAREST:
            getQueue().enqueue(kernel::resize<T, AF_INTERP_NEAREST>, out, in);
            break;
        case AF_INTERP_BILINEAR:
            getQueue().enqueue(kernel::resize<T, AF_INTERP_BILINEAR>, out, in);
            break;
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::resize<T, AF_INTERP_LOWER>, out, in);
            break;
        default: break;
    }
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> resize<T>(const Array<T> &in, const dim_t odim0, \
                                const dim_t odim1,                     \
                                const af_interp_type method);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
