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

        __constant__ float c_tmat[6 * 256];

        template <typename T>
        __host__ __device__
        void calc_affine_inverse(T *txo, const T *txi)
        {
            T det = txi[0]*txi[4] - txi[1]*txi[3];

            txo[0] = txi[4] / det;
            txo[1] = txi[3] / det;
            txo[3] = txi[1] / det;
            txo[4] = txi[0] / det;

            txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
            txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
        }

        ///////////////////////////////////////////////////////////////////////////
        // Transform Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool inverse, af_interp_type method>
        __global__ static void
        transform_kernel(Param<T> out, CParam<T> in, const dim_type nimages,
                         const dim_type ntransforms, const dim_type blocksXPerImage)
        {
            // Compute which image set
            const dim_type setId = blockIdx.x / blocksXPerImage;
            const dim_type blockIdx_x = blockIdx.x - setId * blocksXPerImage;

            // Get thread indices
            const dim_type xx = blockIdx_x * blockDim.x + threadIdx.x;
            const dim_type yy = blockIdx.y * blockDim.y + threadIdx.y;

            const dim_type limages = min(out.dims[2] - setId * nimages, nimages);

            if(xx >= out.dims[0] || yy >= out.dims[1] * ntransforms)
                return;

            // Index of channel of images and transform
            //const dim_type i_idx = xx / out.dims[0];
            const dim_type t_idx = yy / out.dims[1];

            // Index in local channel -> This is output index
            //const dim_type xido = xx - i_idx * out.dims[0];
            const dim_type xido = xx;
            const dim_type yido = yy - t_idx * out.dims[1];

            // Global offset
            //          Offset for transform channel + Offset for image channel.
                  T *optr = out.ptr + t_idx * nimages * out.strides[2] + setId * nimages * out.strides[2];
            const T *iptr = in.ptr  + setId * nimages * in.strides[2];

            // Transform is in constant memory.
            const float *tmat_ptr = c_tmat + t_idx * 6;
            float tmat[6];

            // We expect a inverse transform matrix by default
            // If it is an forward transform, then we need its inverse
            if(inverse) {
                #pragma unroll
                for(int i = 0; i < 6; i++)
                    tmat[i] = tmat_ptr[i];
            } else {
                calc_affine_inverse(tmat, tmat_ptr);
            }

            if (xido >= out.dims[0] && yido >= out.dims[1]) return;

            switch(method) {
                case AF_INTERP_NEAREST:
                    transform_n(optr, out, iptr, in, tmat, xido, yido, limages); break;
                case AF_INTERP_BILINEAR:
                    transform_b(optr, out, iptr, in, tmat, xido, yido, limages); break;
                default: break;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, af_interp_type method>
        void transform(Param<T> out, CParam<T> in, CParam<float> tf,
                       const bool inverse)
        {
            dim_type nimages = in.dims[2];
            // Multiplied in src/backend/transform.cpp
            const dim_type ntransforms = out.dims[2] / in.dims[2];

            // Copy transform to constant memory.
            CUDA_CHECK(cudaMemcpyToSymbol(c_tmat, tf.ptr, ntransforms * 6 * sizeof(float), 0,
                                          cudaMemcpyDeviceToDevice));

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            const dim_type blocksXPerImage = blocks.x;
            if(nimages > TI) {
                dim_type tile_images = divup(nimages, TI);
                nimages = TI;
                blocks.x = blocks.x * tile_images;
            }

            if (ntransforms > 1) { blocks.y *= ntransforms; }

            if(inverse) {
                transform_kernel<T, true, method><<<blocks, threads>>>
                                (out, in, nimages, ntransforms, blocksXPerImage);
            } else {
                transform_kernel<T, false, method><<<blocks, threads>>>
                                (out, in, nimages, ntransforms, blocksXPerImage);
            }
            POST_LAUNCH_CHECK();
        }
    }
}
