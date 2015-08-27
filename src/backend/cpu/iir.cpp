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
#include <err_cpu.hpp>
#include <math.hpp>
#include <arith.hpp>
#include <convolve.hpp>

using af::dim4;

namespace cpu
{
    template<typename T>
    Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x)
    {
        T h_a0 = a.get()[0];
        Array<T> a0 = createValueArray<T>(b.dims(), h_a0);

        ConvolveBatchKind type = x.ndims() == 1 ? CONVOLVE_BATCH_NONE : CONVOLVE_BATCH_SAME;
        if (x.ndims() != b.ndims()) {
            type = (x.ndims() < b.ndims()) ? CONVOLVE_BATCH_KERNEL : CONVOLVE_BATCH_SIGNAL;
        }

        // Extract the first N elements
        Array<T> c = convolve<T, T, 1, true>(x, b, type);
        dim4 cdims = c.dims();
        cdims[0] = x.dims()[0];
        c.resetDims(cdims);

        int num_a = a.dims()[0];

        dim4 ydims = c.dims();
        Array<T> y = createEmptyArray<T>(ydims);

        for (int l = 0; l < (int)ydims[3]; l++) {
            dim_t yidx3 = l * y.strides()[3];
            dim_t cidx3 = l * c.strides()[3];
            dim_t aidx3 = l * a.strides()[3];

            for (int k = 0; k < (int)ydims[2]; k++) {

                dim_t yidx2 = k * y.strides()[2] + yidx3;
                dim_t cidx2 = k * c.strides()[2] + cidx3;
                dim_t aidx2 = k * a.strides()[2] + aidx3;

                for (int j = 0; j < (int)ydims[1]; j++) {

                    dim_t yidx1 = j * y.strides()[1] + yidx2;
                    dim_t cidx1 = j * c.strides()[1] + cidx2;
                    dim_t aidx1 = j * a.strides()[1] + aidx2;

                    std::vector<T> h_z(num_a);

                    const T *h_a = a.get() + (a.ndims() > 1 ? aidx1 : 0);
                    T *h_c = c.get() + cidx1;
                    T *h_y = y.get() + yidx1;

                    for (int i = 0; i < (int)ydims[0]; i++) {

                        T y = h_y[i] = (h_c[i] + h_z[0]) /  h_a[0];
                        for (int ii = 1; ii < num_a; ii++) {
                            h_z[ii - 1] = h_z[ii] - h_a[ii] * y;
                        }
                    }
                }
            }
        }

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
