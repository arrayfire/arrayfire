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
        static const unsigned TX = 16;
        static const unsigned TY = 16;
        // Used for batching images
        static const unsigned TI = 4;

        __constant__ float c_tmat[3072]; // Allows 512 Affine Transforms and 340 Persp. Transforms

        template <typename T>
        __host__ __device__
        void calc_transf_inverse(T *txo, const T *txi, const bool perspective)
        {
            if (perspective) {
                txo[0] =   txi[4]*txi[8] - txi[5]*txi[7];
                txo[1] = -(txi[1]*txi[8] - txi[2]*txi[7]);
                txo[2] =   txi[1]*txi[5] - txi[2]*txi[4];

                txo[3] = -(txi[3]*txi[8] - txi[5]*txi[6]);
                txo[4] =   txi[0]*txi[8] - txi[2]*txi[6];
                txo[5] = -(txi[0]*txi[5] - txi[2]*txi[3]);

                txo[6] =   txi[3]*txi[7] - txi[4]*txi[6];
                txo[7] = -(txi[0]*txi[7] - txi[1]*txi[6]);
                txo[8] =   txi[0]*txi[4] - txi[1]*txi[3];

                T det = txi[0]*txo[0] + txi[1]*txo[3] + txi[2]*txo[6];

                txo[0] /= det; txo[1] /= det; txo[2] /= det;
                txo[3] /= det; txo[4] /= det; txo[5] /= det;
                txo[6] /= det; txo[7] /= det; txo[8] /= det;
                }
            else {
                T det = txi[0]*txi[4] - txi[1]*txi[3];

                txo[0] = txi[4] / det;
                txo[1] = txi[3] / det;
                txo[3] = txi[1] / det;
                txo[4] = txi[0] / det;

                txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
                txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Transform Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool inverse, int order>
        __global__ static void
        transform_kernel(Param<T> out, CParam<T> in,
                         const int nImg2, const int nImg3, const int nTfs2, const int nTfs3,
                         const int batchImg2,
                         const int blocksXPerImage, const int blocksYPerImage,
                         const bool perspective, af_interp_type method)
        {
            // Image Ids
            const int imgId2 = blockIdx.x / blocksXPerImage;
            const int imgId3 = blockIdx.y / blocksYPerImage;

            // Block in local image
            const int blockIdx_x = blockIdx.x - imgId2 * blocksXPerImage;
            const int blockIdx_y = blockIdx.y - imgId3 * blocksYPerImage;

            // Get thread indices in local image
            const int xido = blockIdx_x * blockDim.x + threadIdx.x;
            const int yido = blockIdx_y * blockDim.y + threadIdx.y;

            // Image iteration loop count for image batching
            int limages = min(max(out.dims[2] - imgId2 * nImg2, 1), batchImg2);

            if(xido >= out.dims[0] || yido >= out.dims[1])
                return;

            // Index of transform
            const int eTfs2 = max((nTfs2 / nImg2), 1);
            const int eTfs3 = max((nTfs3 / nImg3), 1);

            int t_idx3 = -1;    // init
            int t_idx2 = -1;    // init
            int t_idx2_offset = 0;

            if(nTfs3 == 1) {
                t_idx3 = 0;     // Always 0 as only 1 transform defined
            } else {
                if(nTfs3 == nImg3) {
                    t_idx3 = imgId3;    // One to one batch with all transforms defined
                } else {
                    t_idx3 = blockIdx.z / eTfs2;    // Transform batched, calculate
                    t_idx2_offset = t_idx3 * nTfs2;
                }
            }

            if(nTfs2 == 1) {
                t_idx2 = 0;     // Always 0 as only 1 transform defined
            } else {
                if(nTfs2 == nImg2) {
                    t_idx2 = imgId2;    // One to one batch with all transforms defined
                } else {
                    t_idx2 = blockIdx.z - t_idx2_offset;   // Transform batched, calculate
                }
            }

            // Linear transform index
            const int t_idx = t_idx2 + t_idx3 * nTfs2;
            int outoff = 0;

            // Global offsets
            const int inoff= imgId2 * batchImg2 * in.strides[2] + imgId3 * in.strides[3];
            if(nImg2 == nTfs2 || nImg2 > 1) {   // One-to-One or Image on dim2
                outoff += imgId2 * batchImg2 * out.strides[2];
            } else {                            // Transform batched on dim2
                outoff += t_idx2 * out.strides[2];
            }

            if(nImg3 == nTfs3 || nImg3 > 1) {   // One-to-One or Image on dim3
                outoff += imgId3 * out.strides[3];
            } else {                            // Transform batched on dim2
                outoff += t_idx3 * out.strides[3];
            }

            // Transform is in constant memory.
            const int transf_len = (perspective ? 9 : 6);
            const float *tmat_ptr = c_tmat + t_idx * transf_len;
            float tmat[9];

            // We expect a inverse transform matrix by default
            // If it is an forward transform, then we need its inverse
            if(inverse) {
                #pragma unroll 3
                for(int i = 0; i < transf_len; i++)
                    tmat[i] = tmat_ptr[i];
            } else {
                calc_transf_inverse(tmat, tmat_ptr, perspective);
            }

            const int loco = outoff + (yido * out.strides[1] + xido);

            // Compute input index
            typedef typename itype_t<T>::wtype WT;
            WT xidi = xido * tmat[0] + yido * tmat[1] + tmat[2];
            WT yidi = xido * tmat[3] + yido * tmat[4] + tmat[5];

            if (perspective) {
                const WT W = xido * tmat[6] + yido * tmat[7] + tmat[8];
                xidi /= W;
                yidi /= W;
            }

            if (xidi < -0.0001 || yidi < -0.0001 || in.dims[0] <= xidi || in.dims[1] <= yidi) {
                for(int i = 0; i < limages; i++) {
                    out.ptr[loco + i * out.strides[2]] = scalar<T>(0.0f);
                }
                return;
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
        void transform(Param<T> out, CParam<T> in, CParam<float> tf,
                       const bool inverse, const bool perspective,
                       af_interp_type method)
        {
            const int nImg2 = in.dims[2];
            const int nImg3 = in.dims[3];
            const int nTfs2 = tf.dims[2];
            const int nTfs3 = tf.dims[3];

            const int tf_len = (perspective) ? 9 : 6;

            // Copy transform to constant memory.
            CUDA_CHECK(cudaMemcpyToSymbolAsync(c_tmat, tf.ptr,
                                               nTfs2 * nTfs3 * tf_len * sizeof(float),
                                               0, cudaMemcpyDeviceToDevice,
                                               cuda::getActiveStream()));

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            const int blocksXPerImage = blocks.x;
            const int blocksYPerImage = blocks.y;

            // Takes care of all types of batching
            // One-to-one batching is only done on blocks.x
            // TODO If dim2 is not one-to-one batched, then divide blocks.x by factor
            int batchImg2 = 1;
            if(nImg2 != nTfs2)
                batchImg2 = min(nImg2, TI);

            blocks.x *= (nImg2 / batchImg2);
            blocks.y *= nImg3;

            // Use blocks.z for transforms
            blocks.z *= max((nTfs2 / nImg2), 1)
                     *  max((nTfs3 / nImg3), 1);

            if(inverse) {
                CUDA_LAUNCH((transform_kernel<T, true, order>), blocks, threads, out, in,
                            nImg2, nImg3, nTfs2, nTfs3, batchImg2,
                            blocksXPerImage, blocksYPerImage,
                            perspective, method);
            } else {
                CUDA_LAUNCH((transform_kernel<T, false, order>), blocks, threads, out, in,
                            nImg2, nImg3, nTfs2, nTfs3, batchImg2,
                            blocksXPerImage, blocksYPerImage,
                            perspective, method);
            }
            POST_LAUNCH_CHECK();
        }
    }
}
