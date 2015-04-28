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

        template<typename To, typename Ti, int dim>
        __global__
        void join_kernel(Param<To> out, CParam<Ti> in,
                         const dim_type o0, const dim_type o1, const dim_type o2, const dim_type o3,
                         const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            const dim_type iz = blockIdx.x / blocksPerMatX;
            const dim_type iw = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - iz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - iw * blocksPerMatY;

            const dim_type xx = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type yy = threadIdx.y + blockIdx_y * blockDim.y;

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            To *d_out = out.ptr;
            Ti const *d_in = in.ptr;

            if(iz < in.dims[2] && iw < in.dims[3]) {
                d_out = d_out + (iz + o2) * out.strides[2] + (iw + o3) * out.strides[3];
                d_in  = d_in  + iz * in.strides[2] + iw * in.strides[3];

                for (dim_type iy = yy; iy < in.dims[1]; iy += incy) {
                    Ti const *d_in_ = d_in + iy * in.strides[1];
                    To *d_out_ = d_out + (iy + o1) * out.strides[1];

                    for (dim_type ix = xx; ix < in.dims[0]; ix += incx) {
                        d_out_[ix + o0] = d_in_[ix];
                    }
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename To, typename Tx, int dim>
        void join(Param<To> out, CParam<Tx> X, const af::dim4 &offset)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(X.dims[0], TILEX);
            dim_type blocksPerMatY = divup(X.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * X.dims[2],
                        blocksPerMatY * X.dims[3],
                        1);

            join_kernel<To, Tx, dim><<<blocks, threads>>>
                       (out, X, offset[0], offset[1], offset[2], offset[3],
                        blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
