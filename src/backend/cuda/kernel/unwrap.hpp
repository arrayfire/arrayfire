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
                           const dim_t px, const dim_t py, const dim_t nx, dim_t repsPerColumn)
        {
            // Compute channel and volume
            const dim_t w = blockIdx.y / in.dims[2];
            const dim_t z = blockIdx.y % in.dims[2];

            if(w >= in.dims[3] || z >= in.dims[2])
                return;

            // Compute offset for channel and volume
            const dim_t cOut = w * out.strides[3] + z * out.strides[2];
            const dim_t cIn  = w *  in.strides[3] + z *  in.strides[2];

            // Compute the output column index
            const dim_t colId = blockIdx.x * blockDim.y + threadIdx.y;

            if(colId >= out.dims[1])
                return;

            // Compute the starting index of window in x and y of input
            const dim_t startx = (colId % nx) * sx;
            const dim_t starty = (colId / nx) * sy;

            const dim_t spx = startx - px;
            const dim_t spy = starty - py;

            // Offset the global pointers to the respective starting indices
                  T* optr = out.ptr + cOut + colId * out.strides[1];
            const T* iptr = in.ptr  + cIn;

            bool cond = false;
            if(spx >= 0 && spx + wx < in.dims[0] && spy >= 0 && spy + wy < in.dims[1])
                cond = true;

            for(int i = 0; i < repsPerColumn; i++) {
                // Compute output index local to column
                const dim_t colIndex = i * threads + threadIdx.x;

                if(colIndex >= out.dims[0])
                    return;

                // Compute input index local to window
                const dim_t x = colIndex % wx;
                const dim_t y = colIndex / wx;

                const dim_t xpad = spx + x;
                const dim_t ypad = spy + y;

                const dim_t outIdx = (y * wx + x) * out.strides[0];

                // Copy
                if(cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 && ypad < in.dims[1])) {
                    const dim_t inIdx = ypad * in.strides[1] + xpad * in.strides[0];
                    optr[outIdx] = iptr[inIdx];
                } else {
                    optr[outIdx] = scalar<T>(0.0);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, int TX>
        void unwrap(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const dim_t nx)
        {
            dim3 threads(TX, 256 / TX, 1);

            dim_t repsPerColumn = 1;
            if(TX == 256 && wx * wy > 256) {
                repsPerColumn = divup((wx * wy), 256);
            }

            dim3 blocks(divup(out.dims[1], threads.y), out.dims[2] * out.dims[3]);

            unwrap_kernel<T, TX><<<blocks, threads>>>(out, in, wx, wy, sx, sy, px, py, nx, repsPerColumn);
            POST_LAUNCH_CHECK();
        }
    }
}

