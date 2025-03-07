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
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T, int dim>
void range(Param<T> output) {
    T* out = output.get();

    const dim4 dims    = output.dims();
    const dim4 strides = output.strides();

    for (dim_t w = 0; w < dims[3]; w++) {
        dim_t offW = w * strides[3];
        for (dim_t z = 0; z < dims[2]; z++) {
            dim_t offWZ = offW + z * strides[2];
            for (dim_t y = 0; y < dims[1]; y++) {
                dim_t offWZY = offWZ + y * strides[1];
                for (dim_t x = 0; x < dims[0]; x++) {
                    dim_t id = offWZY + x;
                    if (dim == 0) {
                        out[id] = x;
                    } else if (dim == 1) {
                        out[id] = y;
                    } else if (dim == 2) {
                        out[id] = z;
                    } else if (dim == 3) {
                        out[id] = w;
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
