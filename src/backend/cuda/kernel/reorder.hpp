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
        void reorder_kernel(Param<T> out, CParam<T> in, const int d0, const int d1,
                            const int d2, const int d3,
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

            const int incy = blocksPerMatY * blockDim.y;
            const int incx = blocksPerMatX * blockDim.x;

            const int rdims[] = {d0, d1, d2, d3};
            const int o_off   = ow * out.strides[3] + oz * out.strides[2];
                  int ids[4]  = {0};
            ids[rdims[3]] = ow;
            ids[rdims[2]] = oz;

            for(int oy = yy; oy < out.dims[1]; oy += incy) {
                ids[rdims[1]] = oy;
                for(int ox = xx; ox < out.dims[0]; ox += incx) {
                    ids[rdims[0]] = ox;

                    const int oIdx = o_off + oy * out.strides[1] + ox;

                    const int iIdx = ids[3] * in.strides[3] + ids[2] * in.strides[2] +
                                          ids[1] * in.strides[1] + ids[0];

                    out.ptr[oIdx] = in.ptr[iIdx];
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void reorder(Param<T> out, CParam<T> in, const dim_t *rdims)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(out.dims[0], TILEX);
            int blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            CUDA_LAUNCH((reorder_kernel<T>), blocks, threads,
                out, in, rdims[0], rdims[1], rdims[2], rdims[3],
                blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
