/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <iir.hpp>
#include <convolve.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/iir.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x)
{
    b.eval();
    a.eval();
    x.eval();

    ConvolveBatchKind type = x.ndims() == 1 ? CONVOLVE_BATCH_NONE : CONVOLVE_BATCH_SAME;
    if (x.ndims() != b.ndims()) {
        type = (x.ndims() < b.ndims()) ? CONVOLVE_BATCH_KERNEL : CONVOLVE_BATCH_SIGNAL;
    }

    // Extract the first N elements
    Array<T> c = convolve<T, T, 1, true>(x, b, type);
    dim4 cdims = c.dims();
    cdims[0] = x.dims()[0];
    c.resetDims(cdims);

    Array<T> y = createEmptyArray<T>(c.dims());

    getQueue().enqueue(kernel::iir<T>, y, c, a);

    return y;
}

#define INSTANTIATE(T)                          \
    template Array<T> iir(const Array<T> &b,    \
                          const Array<T> &a,    \
                          const Array<T> &x);   \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}
