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
#include <cassert>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;
        static const unsigned TILEX = 128;
        static const unsigned TILEY = 32;

        __host__ __device__
        static inline int simple_mod(const int i, const int dim)
        {
            return (i < dim) ? i : (i - dim);
        }

        template<typename T>
        __global__
        void shift_kernel(Param<T> out, CParam<T> in, const int d0, const int d1,
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

            const int iw = simple_mod((ow + d3), out.dims[3]);
            const int iz = simple_mod((oz + d2), out.dims[2]);

            const int o_off = ow * out.strides[3] + oz * out.strides[2];
            const int i_off = iw *  in.strides[3] + iz *  in.strides[2];

            for(int oy = yy; oy < out.dims[1]; oy += incy) {
                const int iy = simple_mod((oy + d1), out.dims[1]);
                for(int ox = xx; ox < out.dims[0]; ox += incx) {
                    const int ix = simple_mod((ox + d0), out.dims[0]);

                    const int oIdx = o_off + oy * out.strides[1] + ox;
                    const int iIdx = i_off + iy *  in.strides[1] + ix;

                    out.ptr[oIdx] = in.ptr[iIdx];
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void shift(Param<T> out, CParam<T> in, const int *sdims)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(out.dims[0], TILEX);
            int blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            int sdims_[4];
            // Need to do this because we are mapping output to input in the kernel
            for(int i = 0; i < 4; i++) {
                // sdims_[i] will always be positive and always [0, oDims[i]].
                // Negative shifts are converted to position by going the other way round
                sdims_[i] = -(sdims[i] % (int)out.dims[i]) + out.dims[i] * (sdims[i] > 0);
                assert(sdims_[i] >= 0 && sdims_[i] <= out.dims[i]);
            }

            shift_kernel<T><<<blocks, threads>>>(out, in, sdims_[0], sdims_[1], sdims_[2], sdims_[3],
                                                 blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
