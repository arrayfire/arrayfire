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
#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "interp.hpp"

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        constexpr unsigned TX = 16;
        constexpr unsigned TY = 16;
        // Used for batching images
        constexpr int TI = 4;

        typedef struct {
            float tmat[6];
        } tmat_t;

        ///////////////////////////////////////////////////////////////////////////
        // Rotate Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, int order>
        __global__ static void
        rotate_kernel(Param<T> out, CParam<T> in, const tmat_t t,
                      const int nimages, const int nbatches,
                      const int blocksXPerImage, const int blocksYPerImage,
                      af_interp_type method)
        {
            // Compute which image set
            const int setId = blockIdx.x / blocksXPerImage;
            const int blockIdx_x = blockIdx.x - setId * blocksXPerImage;

            const int batch = blockIdx.y / blocksYPerImage;
            const int blockIdx_y = blockIdx.y - batch * blocksYPerImage;

            // Get thread indices
            const int xido = blockIdx_x * blockDim.x + threadIdx.x;
            const int yido = blockIdx_y * blockDim.y + threadIdx.y;

            const int limages = min(out.dims[2] - setId * nimages, nimages);

            if(xido >= out.dims[0] || yido >= out.dims[1])
                return;

            // Compute input index
            typedef typename itype_t<T>::wtype WT;
            WT xidi = xido * t.tmat[0] + yido * t.tmat[1] + t.tmat[2];
            WT yidi = xido * t.tmat[3] + yido * t.tmat[4] + t.tmat[5];

            // Global offset
            //          Offset for transform channel + Offset for image channel.
            int outoff =  setId * nimages * out.strides[2] + batch * out.strides[3];
            int inoff  =  setId * nimages * in.strides[2]  + batch * in.strides[3];
            const int loco = outoff + (yido * out.strides[1] + xido);

            if (order > 1) {
                // Special conditions to deal with boundaries for bilinear and bicubic
                // FIXME: Ideally this condition should be removed or be present for all methods
                // But tests are expecting a different behavior for bilinear and nearest
                if (xidi < -0.0001 || yidi < -0.0001 || in.dims[0] < xidi || in.dims[1] < yidi) {
                    for(int i = 0; i < nimages; i++) {
                        out.ptr[loco + i * out.strides[2]] = scalar<T>(0.0f);
                    }
                    return;
                }
            }

            Interp2<T, WT, order> interp;
            // FIXME: Nearest and lower do not do clamping, but other methods do
            // Make it consistent
            bool clamp = order != 1;
            interp(out, loco, in, inoff, xidi, yidi, method, limages, clamp);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, int order>
        void rotate(Param<T> out, CParam<T> in, const float theta, af_interp_type method)
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

            // Rounding error. Anything more than 3 decimal points wont make a diff
            tmat_t t;
            t.tmat[0] = round( c * 1000) / 1000.0f;
            t.tmat[1] = round(-s * 1000) / 1000.0f;
            t.tmat[2] = round(tx * 1000) / 1000.0f;
            t.tmat[3] = round( s * 1000) / 1000.0f;
            t.tmat[4] = round( c * 1000) / 1000.0f;
            t.tmat[5] = round(ty * 1000) / 1000.0f;

            int nimages = in.dims[2];
            int nbatches = in.dims[3];

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            const int blocksXPerImage = blocks.x;
            const int blocksYPerImage = blocks.y;

            if(nimages > TI) {
                int tile_images = divup(nimages, TI);
                nimages = TI;
                blocks.x = blocks.x * tile_images;
            }

            blocks.y = blocks.y * nbatches;

            CUDA_LAUNCH((rotate_kernel<T, order>), blocks, threads,
                        out, in, t, nimages, nbatches,
                        blocksXPerImage, blocksYPerImage, method);

            POST_LAUNCH_CHECK();
        }
    }
}
