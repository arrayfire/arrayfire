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

        template<typename T, bool same_dims>
        __global__
        void lu_split_kernel(Param<T> lower, Param<T> upper, Param<T> in,
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

            T *d_l = lower.ptr;
            T *d_u = upper.ptr;
            T *d_i = in.ptr;

            if(oz < in.dims[2] && ow < in.dims[3]) {
                d_i = d_i + oz * in.strides[2]    + ow * in.strides[3];
                d_l = d_l + oz * lower.strides[2] + ow * lower.strides[3];
                d_u = d_u + oz * upper.strides[2] + ow * upper.strides[3];

                for (int oy = yy; oy < in.dims[1]; oy += incy) {
                    T *Yd_i = d_i + oy * in.strides[1];
                    T *Yd_l = d_l +  oy * lower.strides[1];
                    T *Yd_u = d_u +  oy * upper.strides[1];
                    for (int ox = xx; ox < in.dims[0]; ox += incx) {
                        if(ox > oy) {
                            if(same_dims || oy < lower.dims[1])
                                Yd_l[ox] = Yd_i[ox];
                            if(!same_dims || ox < upper.dims[0])
                                Yd_u[ox] = scalar<T>(0);
                        } else if (oy > ox) {
                            if(same_dims || oy < lower.dims[1])
                                Yd_l[ox] = scalar<T>(0);
                            if(!same_dims || ox < upper.dims[0])
                                Yd_u[ox] = Yd_i[ox];
                        } else if(ox == oy) {
                            if(same_dims || oy < lower.dims[1])
                                Yd_l[ox] = scalar<T>(1.0);
                            if(!same_dims || ox < upper.dims[0])
                                Yd_u[ox] = Yd_i[ox];
                        }
                    }
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void lu_split(Param<T> lower, Param<T> upper, Param<T> in)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(in.dims[0], TILEX);
            int blocksPerMatY = divup(in.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * in.dims[2],
                        blocksPerMatY * in.dims[3],
                        1);

            if(lower.dims[0] == in.dims[0] && lower.dims[1] == in.dims[1]) {
                lu_split_kernel<T, true><<<blocks, threads>>>(lower, upper, in, blocksPerMatX, blocksPerMatY);
            } else {
                lu_split_kernel<T, false><<<blocks, threads>>>(lower, upper, in, blocksPerMatX, blocksPerMatY);
            }
            POST_LAUNCH_CHECK();
        }
    }
}

