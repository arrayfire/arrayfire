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
#include <err_opencl.hpp>
#include <math.hpp>
#include <arith.hpp>
#include <convolve.hpp>
#include <kernel/iir.hpp>

using af::dim4;

namespace opencl
{
    template<typename T>
    Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x)
    {
        try {

            ConvolveBatchKind type = x.ndims() == 1 ? ONE2ONE : MANY2MANY;
            if (x.ndims() != b.ndims()) {
                type = (x.ndims() < b.ndims()) ? ONE2MANY : MANY2ONE;
            }

            // Extract the first N elements
            Array<T> c = convolve<T, T, 1, true>(x, b, type);
            dim4 cdims = c.dims();
            cdims[0] = x.dims()[0];
            c.resetDims(cdims);

            int num_a = a.dims()[0];

            if (num_a == 1) return c;

            dim4 ydims = c.dims();
            Array<T> y = createEmptyArray<T>(ydims);

            if (a.ndims() > 1) {
                kernel::iir<T,  true>(y, c, a);
            } else {
                kernel::iir<T, false>(y, c, a);
            }

            return y;
        } catch (cl::Error &err) {
            CL_TO_AF_ERROR(err);
        }
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
