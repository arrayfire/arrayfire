/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>

namespace cuda {

template<typename T, bool LOOP1, bool LOOP2>
__global__ void memcopy(Param<T> out, CParam<T> in) {
    const int g0 = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    int g1       = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K

    const bool valid = (g0 < (int)in.dims[0]) && (g1 < (int)in.dims[1]);
    if (valid) {
        const int idims3 = in.dims[3];
        in.ptr += g0 * (int)in.strides[0] + g1 * (int)in.strides[1];
        const int istrides2 = in.strides[2];
        const int istrides3 = in.strides[3];
        out.ptr += g0 * (int)out.strides[0] + g1 * (int)out.strides[1];
        const int ostrides2 = out.strides[2];
        const int ostrides3 = out.strides[3];

#if LOOP1
        const int idims1 = in.dims[1];
        const int iinc1  = gridDim.y * (int)in.strides[1];
        const int oinc1  = gridDim.y * (int)out.strides[1];
        do {
#endif

            int g2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
#if LOOP2
            do {
#endif
                // ALWAYS looping!!
                int ioffset          = g2 * istrides2;
                int ooffset          = g2 * ostrides2;
                const int ioffsetEnd = ioffset + idims3 * istrides3;
                do {
                    T val = in.ptr[ioffset];
                    ioffset += istrides3;
                    out.ptr[ooffset] = val;
                    ooffset += ostrides3;
                } while (ioffset != ioffsetEnd);

#if LOOP2
                g2 += gridDim.z;
            } while (g2 < (int)in.dims[2]);
#endif

#if LOOP1
            g1 += gridDim.y;
            in.ptr += iinc1;
            out.ptr += oinc1;
        } while (g1 < idims1);
#endif
    }
}
}  // namespace cuda
