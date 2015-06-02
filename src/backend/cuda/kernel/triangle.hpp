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

        template<typename T, bool is_upper, bool is_unit_diag>
        __global__
        void triangle_kernel(Param<T> r, CParam<T> in,
                             const int blocksPerMatX, const int blocksPerMatY)
        {
            const int oz = blockIdx.x / blocksPerMatX;
            const int ow = blockIdx.y / blocksPerMatY;

            const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const int blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const int xx = threadIdx.x + blockIdx_x * blockDim.x;
            const int yy = threadIdx.y + blockIdx_y * blockDim.y;

            const int incy = blocksPerMatY * blockDim.y;
            const int incx = blocksPerMatX * blockDim.x;

            T *d_r = r.ptr;
            const T *d_i = in.ptr;

            if(oz < r.dims[2] && ow < r.dims[3]) {
                d_i = d_i + oz * in.strides[2]    + ow * in.strides[3];
                d_r = d_r + oz * r.strides[2] + ow * r.strides[3];

                for (int oy = yy; oy < r.dims[1]; oy += incy) {
                    const T *Yd_i = d_i + oy * in.strides[1];
                    T *Yd_r = d_r +  oy * r.strides[1];

                    for (int ox = xx; ox < r.dims[0]; ox += incx) {

                        bool cond = is_upper ? (oy >= ox) : (oy <= ox);
                        bool do_unit_diag  = is_unit_diag && (ox == oy);
                        if(cond) {
                            Yd_r[ox] = do_unit_diag ? scalar<T>(1) : Yd_i[ox];
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
        template<typename T, bool is_upper, bool is_unit_diag>
        void triangle(Param<T> r, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(r.dims[0], TILEX);
            int blocksPerMatY = divup(r.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * r.dims[2],
                        blocksPerMatY * r.dims[3],
                        1);

            triangle_kernel<T, is_upper, is_unit_diag><<<blocks, threads>>>(r, in, blocksPerMatX, blocksPerMatY);

            POST_LAUNCH_CHECK();
        }
    }
}
