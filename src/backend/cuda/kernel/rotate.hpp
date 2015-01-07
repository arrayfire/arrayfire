/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "transform_interp.hpp"

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;
        // Used for batching images
        static const unsigned TI = 4;

        typedef struct {
            float tmat[6];
        } tmat_t;

        ///////////////////////////////////////////////////////////////////////////
        // Rotate Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, af_interp_type method>
        __global__ static void
        rotate_kernel(Param<T> out, CParam<T> in, const tmat_t t,
                      const dim_type nimages, const dim_type nbatches,
                      const dim_type blocksXPerImage, const dim_type blocksYPerImage)
        {
            // Compute which image set
            const dim_type setId = blockIdx.x / blocksXPerImage;
            const dim_type blockIdx_x = blockIdx.x - setId * blocksXPerImage;

            const dim_type batch = blockIdx.y / blocksYPerImage;
            const dim_type blockIdx_y = blockIdx.y - batch * blocksYPerImage;

            // Get thread indices
            const dim_type xx = blockIdx_x * blockDim.x + threadIdx.x;
            const dim_type yy = blockIdx_y * blockDim.y + threadIdx.y;

            const dim_type limages = min(out.dims[2] - setId * nimages, nimages);

            if(xx >= out.dims[0] || yy >= out.dims[1])
                return;

            // Global offset
            //          Offset for transform channel + Offset for image channel.
                  T *optr = out.ptr + setId * nimages * out.strides[2] + batch * out.strides[3];
            const T *iptr = in.ptr  + setId * nimages * in.strides[2]  + batch * in.strides[3];

            switch(method) {
                case AF_INTERP_NEAREST:
                    transform_n(optr, out, iptr, in, t.tmat, xx, yy, limages); break;
                case AF_INTERP_BILINEAR:
                    transform_b(optr, out, iptr, in, t.tmat, xx, yy, limages); break;
                default: break;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, af_interp_type method>
        void rotate(Param<T> out, CParam<T> in, const float theta)
        {
            const float c = cos(-theta), s = sin(-theta);
            float tx, ty;
            {
                const float nx = 0.5 * (in.dims[0] - 1);
                const float ny = 0.5 * (in.dims[1] - 1);
                const float mx = 0.5 * (out.dims[0] - 1);
                const float my = 0.5 * (out.dims[1] - 1);
                const float sx = (mx * c + my *-s);
                const float sy = (mx * s + my * c);
                tx = -(sx - nx);
                ty = -(sy - ny);
            }

            tmat_t t;
            t.tmat[0] =  c;
            t.tmat[1] = -s;
            t.tmat[2] = tx;
            t.tmat[3] =  s;
            t.tmat[4] =  c;
            t.tmat[5] = ty;

            dim_type nimages = in.dims[2];
            dim_type nbatches = in.dims[3];

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            const dim_type blocksXPerImage = blocks.x;
            const dim_type blocksYPerImage = blocks.y;

            if(nimages > TI) {
                dim_type tile_images = divup(nimages, TI);
                nimages = TI;
                blocks.x = blocks.x * tile_images;
            }

            blocks.y = blocks.y * nbatches;

            rotate_kernel<T, method><<<blocks, threads>>> (out, in, t, nimages, nbatches,
                                    blocksXPerImage, blocksYPerImage);

            POST_LAUNCH_CHECK();
        }
    }
}

