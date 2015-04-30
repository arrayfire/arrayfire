/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;
        static const unsigned TILEX = 128;
        static const unsigned TILEY = 32;

        template<typename T, bool is_upper>
        __global__
        void triangle_kernel(Param<T> r, CParam<T> in,
                             const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            const dim_type oz = blockIdx.x / blocksPerMatX;
            const dim_type ow = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const dim_type xx = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type yy = threadIdx.y + blockIdx_y * blockDim.y;

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            T *d_r = r.ptr;
            const T *d_i = in.ptr;

            if(oz < r.dims[2] && ow < r.dims[3]) {
                d_i = d_i + oz * in.strides[2]    + ow * in.strides[3];
                d_r = d_r + oz * r.strides[2] + ow * r.strides[3];

                for (dim_type oy = yy; oy < r.dims[1]; oy += incy) {
                    const T *Yd_i = d_i + oy * in.strides[1];
                    T *Yd_r = d_r +  oy * r.strides[1];

                    for (dim_type ox = xx; ox < r.dims[0]; ox += incx) {

                        if(!is_upper ^ (oy >= ox)) {
                            Yd_r[ox] = Yd_i[ox];
                        } else {
                            Yd_r[ox] = scalar<T>(0);
                        }
                    }
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool is_upper>
        void triangle(Param<T> r, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(r.dims[0], TILEX);
            dim_type blocksPerMatY = divup(r.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * r.dims[2],
                        blocksPerMatY * r.dims[3],
                        1);

            triangle_kernel<T, is_upper><<<blocks, threads>>>(r, in, blocksPerMatX, blocksPerMatY);

            POST_LAUNCH_CHECK();
        }
    }
}
