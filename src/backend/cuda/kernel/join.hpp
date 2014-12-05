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

            if(xx >= out.dims[0] ||
               yy >= out.dims[1] ||
               oz >= out.dims[2] ||
               ow >= out.dims[3])
                return;

            dim_type odx[] = {xx, yy, oz, ow};
            dim_type idx[] = {xx, yy, oz, ow};

            // These if(dim == <dimensions>) conditions are used to check which array
            // (X or Y) to use. 3 out of the 4 if conditions will not be executed
            // since the kernel is templated.
            // These if-conds decide whether to use X or Y based on the output index
            // They also compute the corrent input index if Y is chosen
            Tx const *in = X.ptr;
            dim_type *str = X.strides;
            if(dim == 2) {
                if(odx[2] >= X.dims[2]) {
                    in = Y.ptr;
                    str = Y.strides;
                    idx[2] = odx[2] - offset[2];
                }
            } else if (dim == 3) {
                if(odx[3] >= X.dims[3]) {
                    in = Y.ptr;
                    str = Y.strides;
                    idx[3] = odx[3] - offset[3];
                }
            }

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            const dim_type ozw = odx[3] * out.strides[3] + odx[2] * out.strides[2];

            for(dim_type oy = yy; oy < out.dims[1]; oy += incy) {
                odx[1] = oy;
                idx[1] = oy;
                if(dim == 1) {
                    in = X.ptr;
                    str = X.strides;
                    if(odx[1] >= X.dims[1]) {
                        in = Y.ptr;
                        str = Y.strides;
                        idx[1] = odx[1] - offset[1];
                    }
                }
                for(dim_type ox = xx; ox < out.dims[0]; ox += incx) {
                    odx[0] = ox;
                    idx[0] = ox;
                    if(dim == 0) {
                        in = X.ptr;
                        str = X.strides;
                        if(odx[0] >= X.dims[0]) {
                            in = Y.ptr;
                            str = Y.strides;
                            idx[0] = odx[0] - offset[0];
                        }
                    }

                    const dim_type izw = idx[3] * str[3] + idx[2] * str[2];
                    dim_type iMem = izw + idx[1] * str[1] + idx[0];
                    dim_type oMem = ozw + odx[1] * out.strides[1] + odx[0];

                    out.ptr[oMem] = in[iMem];
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
