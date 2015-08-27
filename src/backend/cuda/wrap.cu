/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <wrap.hpp>
#include <stdexcept>
#include <err_cuda.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <kernel/wrap.hpp>

namespace cuda
{

    template<typename T>
    Array<T> wrap(const Array<T> &in,
                  const dim_t ox, const dim_t oy,
                  const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py,
                  const bool is_column)
    {
        af::dim4 idims = in.dims();
        af::dim4 odims(ox, oy, idims[2], idims[3]);
        Array<T> out = createValueArray<T>(odims, scalar<T>(0));

        kernel::wrap<T>(out, in, wx, wy, sx, sy, px, py, is_column);
        return out;
    }


#define INSTANTIATE(T)                                          \
    template Array<T> wrap<T> (const Array<T> &in,              \
                               const dim_t ox, const dim_t oy,  \
                               const dim_t wx, const dim_t wy,  \
                               const dim_t sx, const dim_t sy,  \
                               const dim_t px, const dim_t py,  \
                               const bool is_column);


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
}
