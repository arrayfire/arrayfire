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

namespace arrayfire {
namespace cpu {
namespace kernel {

af::dim4 calcOffset(const af::dim4 dims, int dim) {
    af::dim4 offset;
    offset[0] = (dim == 0) ? dims[0] : 0;
    offset[1] = (dim == 1) ? dims[1] : 0;
    offset[2] = (dim == 2) ? dims[2] : 0;
    offset[3] = (dim == 3) ? dims[3] : 0;
    return offset;
}

template<typename T>
void join_append(T *out, const T *X, const af::dim4 &offset,
                 const af::dim4 &xdims, const af::dim4 &ost,
                 const af::dim4 &xst) {
    for (dim_t ow = 0; ow < xdims[3]; ow++) {
        const dim_t xW = ow * xst[3];
        const dim_t oW = (ow + offset[3]) * ost[3];

        for (dim_t oz = 0; oz < xdims[2]; oz++) {
            const dim_t xZW = xW + oz * xst[2];
            const dim_t oZW = oW + (oz + offset[2]) * ost[2];

            for (dim_t oy = 0; oy < xdims[1]; oy++) {
                const dim_t xYZW = xZW + oy * xst[1];
                const dim_t oYZW = oZW + (oy + offset[1]) * ost[1];

                memcpy(out + oYZW + offset[0], X + xYZW, xdims[0] * sizeof(T));
            }
        }
    }
}

template<typename T>
void join(const int dim, Param<T> out, const std::vector<CParam<T>> inputs,
          int n_arrays) {
    af::dim4 zero(0, 0, 0, 0);
    af::dim4 d = zero;
    join_append<T>(out.get(), inputs[0].get(), zero, inputs[0].dims(),
                   out.strides(), inputs[0].strides());
    for (int i = 1; i < n_arrays; i++) {
        d += inputs[i - 1].dims();
        join_append<T>(out.get(), inputs[i].get(), calcOffset(d, dim),
                       inputs[i].dims(), out.strides(), inputs[i].strides());
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
