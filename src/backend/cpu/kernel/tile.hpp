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
void tile(Param<T> out, CParam<T> in) {
    T* outPtr      = out.get();
    const T* inPtr = in.get();

    const af::dim4 iDims = in.dims();
    const af::dim4 oDims = out.dims();
    const af::dim4 ist   = in.strides();
    const af::dim4 ost   = out.strides();

    for (dim_t ow = 0; ow < oDims[3]; ow++) {
        const dim_t iw = ow % iDims[3];
        const dim_t iW = iw * ist[3];
        const dim_t oW = ow * ost[3];
        for (dim_t oz = 0; oz < oDims[2]; oz++) {
            const dim_t iz  = oz % iDims[2];
            const dim_t iZW = iW + iz * ist[2];
            const dim_t oZW = oW + oz * ost[2];
            for (dim_t oy = 0; oy < oDims[1]; oy++) {
                const dim_t iy   = oy % iDims[1];
                const dim_t iYZW = iZW + iy * ist[1];
                const dim_t oYZW = oZW + oy * ost[1];
                for (dim_t ox = 0; ox < oDims[0]; ox++) {
                    const dim_t ix   = ox % iDims[0];
                    const dim_t iMem = iYZW + ix;
                    const dim_t oMem = oYZW + ox;
                    outPtr[oMem]     = inPtr[iMem];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
