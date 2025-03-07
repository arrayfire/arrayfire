/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <math.hpp>

#include <af/dim4.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void diagCreate(Param<T> out, CParam<T> in, int const num) {
    int batch = in.dims(1);
    int size  = out.dims(0);

    T const *iptr = in.get();
    T *optr       = out.get();

    for (int k = 0; k < batch; k++) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                T val = scalar<T>(0);
                if (i == j - num) { val = (num > 0) ? iptr[i] : iptr[j]; }
                optr[i + j * out.strides(1)] = val;
            }
        }
        optr += out.strides(2);
        iptr += in.strides(1);
    }
}

template<typename T>
void diagExtract(Param<T> out, CParam<T> in, int const num) {
    af::dim4 const odims = out.dims();
    af::dim4 const idims = in.dims();

    int const i_off = (num > 0) ? (num * in.strides(1)) : (-num);

    for (int l = 0; l < (int)odims[3]; l++) {
        for (int k = 0; k < (int)odims[2]; k++) {
            const T *iptr =
                in.get() + l * in.strides(3) + k * in.strides(2) + i_off;
            T *optr = out.get() + l * out.strides(3) + k * out.strides(2);

            for (int i = 0; i < (int)odims[0]; i++) {
                T val = scalar<T>(0);
                if (i < idims[0] && i < idims[1])
                    val = iptr[i * in.strides(1) + i];
                optr[i] = val;
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
