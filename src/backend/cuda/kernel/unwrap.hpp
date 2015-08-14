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
#include "config.hpp"

namespace cuda
{
    namespace kernel
    {
        ///////////////////////////////////////////////////////////////////////////
        // Unwrap Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool is_column>
        __global__
        void unwrap_kernel(Param<T> out, CParam<T> in,
                           const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
                           const dim_t px, const dim_t py, const dim_t nx, dim_t reps)
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
            const dim_t id = is_column ?
                (blockIdx.x * blockDim.y + threadIdx.y) :
                (blockIdx.x * blockDim.x + threadIdx.x);

            if (id >= (is_column ? out.dims[1] : out.dims[0])) return;

            // Compute the starting index of window in x and y of input
            const dim_t startx = (id % nx) * sx;
            const dim_t starty = (id / nx) * sy;

            const dim_t spx = startx - px;
            const dim_t spy = starty - py;

            // Offset the global pointers to the respective starting indices
            T* optr = out.ptr + cOut + id * (is_column ? out.strides[1] : 1);
            const T* iptr = in.ptr  + cIn;

            bool cond = (spx >= 0 && spx + wx < in.dims[0] && spy >= 0 && spy + wy < in.dims[1]);

            for(int i = 0; i < reps; i++) {

                // Compute output index local to column
                const dim_t outIdx = is_column ?
                    (i * blockDim.x + threadIdx.x) :
                    (i * blockDim.y + threadIdx.y);

                if(outIdx >= (is_column ? out.dims[0] : out.dims[1]))
                    return;

                // Compute input index local to window
                const dim_t x = outIdx % wx;
                const dim_t y = outIdx / wx;

                const dim_t xpad = spx + x;
                const dim_t ypad = spy + y;

                // Copy
                T val = scalar<T>(0.0);
                if(cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 && ypad < in.dims[1])) {
                    const dim_t inIdx = ypad * in.strides[1] + xpad;
                    val = iptr[inIdx];
                }

                if (is_column) {
                    optr[outIdx] = val;
                } else {
                    optr[outIdx * out.strides[1]] = val;
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T>
        void unwrap_col(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                        const dim_t sx, const dim_t sy,
                        const dim_t px, const dim_t py, const dim_t nx)
        {
            dim_t TX = std::min(THREADS_PER_BLOCK, nextpow2(out.dims[0]));

            dim3 threads(TX, THREADS_PER_BLOCK / TX);
            dim3 blocks(divup(out.dims[1], threads.y), out.dims[2] * out.dims[3]);

            dim_t reps = divup((wx * wy), threads.x); // is > 1 only when TX == 256 && wx * wy > 256

            CUDA_LAUNCH((unwrap_kernel<T, true>), blocks, threads,
                        out, in, wx, wy, sx, sy, px, py, nx, reps);

            POST_LAUNCH_CHECK();
        }

        template<typename T>
        void unwrap_row(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                        const dim_t sx, const dim_t sy,
                        const dim_t px, const dim_t py, const dim_t nx)
        {
            dim3 threads(THREADS_X, THREADS_Y);
            dim3 blocks(divup(out.dims[0], threads.x), out.dims[2] * out.dims[3]);

            dim_t reps = divup((wx * wy), threads.y);

            CUDA_LAUNCH((unwrap_kernel<T, false>), blocks, threads,
                        out, in, wx, wy, sx, sy, px, py, nx, reps);

            POST_LAUNCH_CHECK();
        }

        template <typename T>
        void unwrap(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy,
                    const dim_t px, const dim_t py, const dim_t nx, const bool is_column)
        {

            if (is_column) {
                unwrap_col<T>(out, in, wx, wy, sx, sy, px, py, nx);
            } else {
                unwrap_row<T>(out, in, wx, wy, sx, sy, px, py, nx);
            }
        }

    }
}
