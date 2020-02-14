/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <kernel/wrap.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <wrap.hpp>

using common::half;

namespace cpu {

template<typename T>
void wrap(Array<T> &out, const Array<T> &in, const dim_t ox, const dim_t oy,
          const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
          const dim_t px, const dim_t py, const bool is_column) {
    evalMultiple<T>(std::vector<Array<T> *>{const_cast<Array<T> *>(&in), &out});

    if (is_column) {
        getQueue().enqueue(kernel::wrap_dim<T, 1>, out, in, wx, wy, sx, sy, px,
                           py);
    } else {
        getQueue().enqueue(kernel::wrap_dim<T, 0>, out, in, wx, wy, sx, sy, px,
                           py);
    }
}

#define INSTANTIATE(T)                                                        \
    template void wrap<T>(Array<T> & out, const Array<T> &in, const dim_t ox, \
                          const dim_t oy, const dim_t wx, const dim_t wy,     \
                          const dim_t sx, const dim_t sy, const dim_t px,     \
                          const dim_t py, const bool is_column);

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
#undef INSTANTIATE

template<typename T>
Array<T> wrap_dilated(const Array<T> &in, const dim_t ox, const dim_t oy,
                      const dim_t wx, const dim_t wy, const dim_t sx,
                      const dim_t sy, const dim_t px, const dim_t py,
                      const dim_t dx, const dim_t dy, const bool is_column) {
    af::dim4 idims = in.dims();
    af::dim4 odims(ox, oy, idims[2], idims[3]);

    Array<T> out = createValueArray<T>(odims, scalar<T>(0));
    out.eval();
    in.eval();

    getQueue().enqueue(kernel::wrap_dim_dilated<T>, out, in, wx, wy, sx, sy, px,
                       py, dx, dy, is_column);

    return out;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> wrap_dilated<T>(                                      \
        const Array<T> &in, const dim_t ox, const dim_t oy, const dim_t wx, \
        const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px,     \
        const dim_t py, const dim_t dx, const dim_t dy, const bool is_column);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cpu
