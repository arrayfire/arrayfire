/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <wrap.hpp>

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <kernel/wrap.hpp>

#include <stdexcept>

namespace cuda {

template<typename T>
void wrap(Array<T> &out, const Array<T> &in, const dim_t ox, const dim_t oy,
          const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
          const dim_t px, const dim_t py, const bool is_column) {
    kernel::wrap<T>(out, in, wx, wy, sx, sy, px, py, is_column);
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
}  // namespace cuda
