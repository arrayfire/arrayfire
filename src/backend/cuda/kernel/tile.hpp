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
        void tile_kernel(Param<T> out, CParam<T> in,
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

            const int iz = oz % in.dims[2];
            const int iw = ow % in.dims[3];
            const int izw = iw * in.strides[3] + iz * in.strides[2];
            const int ozw = ow * out.strides[3] + oz * out.strides[2];

            const int incy = blocksPerMatY * blockDim.y;
            const int incx = blocksPerMatX * blockDim.x;

            for(int oy = yy; oy < out.dims[1]; oy += incy) {
                const int iy = oy % in.dims[1];
                for(int ox = xx; ox < out.dims[0]; ox += incx) {
                    const int ix = ox % in.dims[0];

                    int iMem = izw + iy * in.strides[1] + ix;
                    int oMem = ozw + oy * out.strides[1] + ox;

                    out.ptr[oMem] = in.ptr[iMem];
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void tile(Param<T> out, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(out.dims[0], TILEX);
            int blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            CUDA_LAUNCH((tile_kernel<T>), blocks, threads, out, in, blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
