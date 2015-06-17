/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>

namespace cuda
{
    namespace kernel
    {
        ///////////////////////////////////////////////////////////////////////////
        // Resize Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, int threads>
        __global__
        void unwrap_kernel(Param<T> out, CParam<T> in,
                           const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
                           dim_t repsPerColumn)
        {
            const dim_t w = blockIdx.y / in.dims[2];
            const dim_t z = blockIdx.y % in.dims[2];

            if(w >= in.dims[3] || z >= in.dims[2])
                return;

            const dim_t cOut = w * out.strides[3] + z * out.strides[2];
            const dim_t cIn  = w *  in.strides[3] + z *  in.strides[2];

            const dim_t nx = (in.dims[0] - wx) / sx + 1;
            //dim_t ny = (in.dims[1] - wy) / sy + 1;

            const dim_t colId = blockIdx.x * blockDim.y + threadIdx.y;

            if(colId >= out.dims[1])
                return;

            const dim_t startx = (colId % nx) * sx;
            const dim_t starty = (colId / nx) * sy;

                  T* optr = out.ptr + cOut + colId * out.strides[1];
            const T* iptr = in.ptr  + cIn  + starty * in.strides[1] + startx;

            for(int i = 0; i < repsPerColumn; i++) {
                const dim_t colIndex = i * threads + threadIdx.x;

                if(colIndex >= out.dims[0])
                    return;

                const dim_t x = colIndex % wx;
                const dim_t y = colIndex / wx;

                const dim_t outIdx = (y * wx + x) * out.strides[0];
                const dim_t inIdx = y * in.strides[1] + x * in.strides[0];

                optr[outIdx] = iptr[inIdx];
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, int TX>
        void unwrap(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy)
        {
            dim3 threads(TX, 256 / TX, 1);

            dim_t repsPerColumn = 1;
            if(TX == 256 && wx * wy > 256) {
                repsPerColumn = (wx * wy) / 256;
            }

            dim3 blocks(divup(out.dims[1], threads.y), out.dims[2] * out.dims[3]);

            unwrap_kernel<T, TX><<<blocks, threads>>>(out, in, wx, wy, sx, sy, repsPerColumn);
            POST_LAUNCH_CHECK();
        }
    }
}

