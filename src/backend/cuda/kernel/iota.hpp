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
        void iota_kernel(Param<T> out,
                         const int s0, const int s1, const int s2, const int s3,
                         const int t0, const int t1, const int t2, const int t3,
                         const int blocksPerMatX, const int blocksPerMatY)
        {
            const int oz = blockIdx.x / blocksPerMatX;
            const int ow = blockIdx.y / blocksPerMatY;

            const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const int blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const int xx = threadIdx.x + blockIdx_x * blockDim.x;
            const int yy = threadIdx.y + blockIdx_y * blockDim.y;

            if(xx >= out.dims[0] ||
               yy >= out.dims[1] ||
               oz >= out.dims[2] ||
               ow >= out.dims[3])
                return;

            const int ozw = ow * out.strides[3] + oz * out.strides[2];

            T val = (ow % s3) * s2 * s1 * s0;
            val  += (oz % s2) * s1 * s0;

            const int incy = blocksPerMatY * blockDim.y;
            const int incx = blocksPerMatX * blockDim.x;

            for(int oy = yy; oy < out.dims[1]; oy += incy) {
                int oyzw = ozw + oy * out.strides[1];
                T valY = val + (oy % s1) * s0;
                for(int ox = xx; ox < out.dims[0]; ox += incx) {
                    int oidx = oyzw + ox;

                    out.ptr[oidx] = valY + (ox % s0);
                }
            }
        }


        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void iota(Param<T> out, const dim4 &sdims, const dim4 &tdims)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(out.dims[0], TILEX);
            int blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            CUDA_LAUNCH((iota_kernel<T>), blocks, threads,
                    out, sdims[0], sdims[1], sdims[2], sdims[3],
                    tdims[0], tdims[1], tdims[2], tdims[3], blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
