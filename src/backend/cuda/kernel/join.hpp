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
                         const int o0, const int o1, const int o2, const int o3,
                         const int blocksPerMatX, const int blocksPerMatY)
        {
            const int iz = blockIdx.x / blocksPerMatX;
            const int iw = blockIdx.y / blocksPerMatY;

            const int blockIdx_x = blockIdx.x - iz * blocksPerMatX;
            const int blockIdx_y = blockIdx.y - iw * blocksPerMatY;

            const int xx = threadIdx.x + blockIdx_x * blockDim.x;
            const int yy = threadIdx.y + blockIdx_y * blockDim.y;

            const int incy = blocksPerMatY * blockDim.y;
            const int incx = blocksPerMatX * blockDim.x;

            To *d_out = out.ptr;
            Ti const *d_in = in.ptr;

            if(iz < in.dims[2] && iw < in.dims[3]) {
                d_out = d_out + (iz + o2) * out.strides[2] + (iw + o3) * out.strides[3];
                d_in  = d_in  + iz * in.strides[2] + iw * in.strides[3];

                for (int iy = yy; iy < in.dims[1]; iy += incy) {
                    Ti const *d_in_ = d_in + iy * in.strides[1];
                    To *d_out_ = d_out + (iy + o1) * out.strides[1];

                    for (int ix = xx; ix < in.dims[0]; ix += incx) {
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

            int blocksPerMatX = divup(X.dims[0], TILEX);
            int blocksPerMatY = divup(X.dims[1], TILEY);
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
