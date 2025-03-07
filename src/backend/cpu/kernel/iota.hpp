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

template<typename T>
void iota(Param<T> output, const af::dim4& sdims) {
    const af::dim4 dims    = output.dims();
    data_t<T>* out         = output.get();
    const af::dim4 strides = output.strides();

    for (dim_t w = 0; w < dims[3]; w++) {
        dim_t offW = w * strides[3];
        dim_t valW = (w % sdims[3]) * sdims[0] * sdims[1] * sdims[2];
        for (dim_t z = 0; z < dims[2]; z++) {
            dim_t offWZ = offW + z * strides[2];
            dim_t valZ  = valW + (z % sdims[2]) * sdims[0] * sdims[1];
            for (dim_t y = 0; y < dims[1]; y++) {
                dim_t offWZY = offWZ + y * strides[1];
                dim_t valY   = valZ + (y % sdims[1]) * sdims[0];
                for (dim_t x = 0; x < dims[0]; x++) {
                    dim_t id = offWZY + x;
                    out[id]  = valY + (x % sdims[0]);
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
