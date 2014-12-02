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

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

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

        template<typename T>
        __device__
        void transform_n(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const dim_type xido, const dim_type yido)
        {
            // Compute input index
            dim_type xidi = round(xido * tmat[0]
                                + yido * tmat[1]
                                       + tmat[2]);
            dim_type yidi = round(xido * tmat[3]
                                + yido * tmat[4]
                                       + tmat[5]);

            // Makes scale give same output as resize
            // But fails rotate tests
            //if (xidi >= in.dims[0]) { xidi = in.dims[0] - 1; }
            //if (yidi >= in.dims[1]) { yidi = in.dims[1] - 1; }

            // Compute memory location of indices
            dim_type loci = (yidi * in.strides[1]  + xidi);
            dim_type loco = (yido * out.strides[1] + xido);

            // Copy to output
            T val = scalar<T>(0);

            if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = iptr[loci];

            optr[loco] = val;
        }

        template<typename T>
        __device__
        void transform_b(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const dim_type xido, const dim_type yido)
        {
            dim_type loco = (yido * out.strides[1] + xido);

            // Compute input index
            const float xidi = xido * tmat[0]
                             + yido * tmat[1]
                                    + tmat[2];
            const float yidi = xido * tmat[3]
                             + yido * tmat[4]
                                    + tmat[5];

            if (xidi < 0 || yidi < 0 || in.dims[0] < xidi || in.dims[1] < yidi) {
                optr[loco] = scalar<T>(0.0f);
                return;
            }

            const float grd_x = floor(xidi),  grd_y = floor(yidi);
            const float off_x = xidi - grd_x, off_y = yidi - grd_y;

            dim_type ioff = grd_y * in.strides[1] + grd_x;

            // Check if pVal and pVal + 1 are both valid indices
            bool condY = (yidi < in.dims[1] - 1);
            bool condX = (xidi < in.dims[0] - 1);

            // Compute weights used
            float wt00 = (1.0 - off_x) * (1.0 - off_y);
            float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
            float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
            float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

            float wt = wt00 + wt10 + wt01 + wt11;

            // Compute Weighted Values
            T zero = scalar<T>(0.0f);
            T v00 =                    wt00 * iptr[ioff];
            T v10 = (condY) ?          wt10 * iptr[ioff + in.strides[1]]     : zero;
            T v01 = (condX) ?          wt01 * iptr[ioff + 1]                 : zero;
            T v11 = (condX && condY) ? wt11 * iptr[ioff + in.strides[1] + 1] : zero;
            T vo = v00 + v10 + v01 + v11;

            optr[loco] = (vo / wt);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Transform Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool inverse, af_interp_type method>
        __global__ static void
        transform_kernel(Param<T> out, CParam<T> in,
                         const dim_type nimages, const dim_type ntransforms)
        {
            // Get thread indices
            const dim_type xx = blockIdx.x * blockDim.x + threadIdx.x;
            const dim_type yy = blockIdx.y * blockDim.y + threadIdx.y;

            if(xx >= out.dims[0] * nimages || yy >= out.dims[1] * ntransforms)
                return;

            // Index of channel of images and transform
            const dim_type i_idx = xx / out.dims[0];
            const dim_type t_idx = yy / out.dims[1];

            // Index in local channel -> This is output index
            const dim_type xido = xx - i_idx * out.dims[0];
            const dim_type yido = yy - t_idx * out.dims[1];

            // Global offset
            //          Offset for transform channel + Offset for image channel.
            T *optr = out.ptr + t_idx * nimages * out.strides[2] + i_idx * out.strides[2];
            const T *iptr = in.ptr + i_idx * in.strides[2];

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
                    transform_n(optr, out, iptr, in, tmat, xido, yido); break;
                case AF_INTERP_BILINEAR:
                    transform_b(optr, out, iptr, in, tmat, xido, yido); break;
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
            const dim_type nimages = in.dims[2];
            // Multiplied in src/backend/transform.cpp
            const dim_type ntransforms = out.dims[2] / in.dims[2];

            // Copy transform to constant memory.
            CUDA_CHECK(cudaMemcpyToSymbol(c_tmat, tf.ptr, ntransforms * 6 * sizeof(float), 0,
                                          cudaMemcpyDeviceToDevice));

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            if (nimages > 1)     { blocks.x *= nimages;   }
            if (ntransforms > 1) { blocks.y *= ntransforms; }

            if(inverse) {
                transform_kernel<T, true, method><<<blocks, threads>>>
                                (out, in, nimages, ntransforms);
            } else {
                transform_kernel<T, false, method><<<blocks, threads>>>
                                (out, in, nimages, ntransforms);
            }
            POST_LAUNCH_CHECK();
        }
    }
}
