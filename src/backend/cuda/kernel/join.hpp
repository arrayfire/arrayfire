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
        static const unsigned TILEX = 256;
        static const unsigned TILEY = 32;

        template<typename Tx, typename Ty, int dim>
        __global__
        void join_kernel(Param<Tx> out, CParam<Tx> X, CParam<Ty> Y,
                         const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            dim_type offset[4];
            offset[0] = (dim == 0) ? X.dims[0] : 0;
            offset[1] = (dim == 1) ? X.dims[1] : 0;
            offset[2] = (dim == 2) ? X.dims[2] : 0;
            offset[3] = (dim == 3) ? X.dims[3] : 0;

            const dim_type oz = blockIdx.x / blocksPerMatX;
            const dim_type ow = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const dim_type xx = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type yy = threadIdx.y + blockIdx_y * blockDim.y;

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            Tx *d_out = out.ptr;
            Tx const *d_X = X.ptr;
            Ty const *d_Y = Y.ptr;

            if(oz < out.dims[2] && ow < out.dims[3]) {
                d_out = d_out + oz * out.strides[2] + ow * out.strides[3];
                d_X = d_X + oz * X.strides[2] + ow * X.strides[3];
                d_Y = d_Y + (oz - offset[2]) * Y.strides[2] + (ow - offset[3]) * Y.strides[3];
                bool cond2 = oz < X.dims[2] && ow < X.dims[3];

                for (dim_type oy = yy; oy < out.dims[1]; oy += incy) {
                    bool cond1 = cond2 && oy < X.dims[1];
                    Tx const *d_X_ = d_X + oy * X.strides[1];
                    Ty const *d_Y_ = d_Y + (oy - offset[1]) * Y.strides[1];
                    Tx *d_out_ = d_out + oy * out.strides[1];

                    for (dim_type ox = xx; ox < out.dims[0]; ox += incx) {
                        bool cond0 = cond1 && ox < X.dims[0];
                        d_out_[ox] = cond0 ? d_X_[ox] : d_Y_[ox - offset[0]];
                    }
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename Tx, typename Ty, int dim>
        void join(Param<Tx> out, CParam<Tx> X, CParam<Ty> Y)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(out.dims[0], TILEX);
            dim_type blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            join_kernel<Tx, Ty, dim><<<blocks, threads>>>(out, X, Y, blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
