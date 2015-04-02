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
        static const unsigned TILEX = 512;
        static const unsigned TILEY = 32;

        template<typename T>
        __global__
        void range_kernel(Param<T> out, const int dim,
                          const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            const int mul0 = (dim == 0);
            const int mul1 = (dim == 1);
            const int mul2 = (dim == 2);
            const int mul3 = (dim == 3);

            const dim_type oz = blockIdx.x / blocksPerMatX;
            const dim_type ow = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const dim_type xx = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type yy = threadIdx.y + blockIdx_y * blockDim.y;

            if(xx >= out.dims[0] ||
               yy >= out.dims[1] ||
               oz >= out.dims[2] ||
               ow >= out.dims[3])
                return;

            const dim_type ozw = ow * out.strides[3] + oz * out.strides[2];

            T valZW = (mul3 * ow) + (mul2 * oz);

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            for(dim_type oy = yy; oy < out.dims[1]; oy += incy) {
                T valYZW = valZW + (mul1 * oy);
                dim_type oyzw = ozw + oy * out.strides[1];
                for(dim_type ox = xx; ox < out.dims[0]; ox += incx) {
                    dim_type oidx = oyzw + ox;
                    T val = valYZW + (ox * mul0);

                    out.ptr[oidx] = val;
                }
            }
        }


        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void range(Param<T> out, const int dim)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(out.dims[0], TILEX);
            dim_type blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            range_kernel<T><<<blocks, threads>>>(out, dim, blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
