/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, bool isHSV2RGB>
__global__
void convert(Param<T> out, CParam<T> in, int nBBS)
{
    // batch offsets
    unsigned batchId= blockIdx.x / nBBS;
    const T* src    = (const T *) in.ptr + (batchId *  in.strides[3]);
    T*       dst    = (T *      )out.ptr + (batchId * out.strides[3]);
    // global indices
    int gx = blockDim.x * (blockIdx.x-batchId*nBBS) + threadIdx.x;
    int gy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gx < out.dims[0] && gy < out.dims[1]) {

        int oIdx0 = gx + gy * out.strides[1];
        int oIdx1 = oIdx0 + out.strides[2];
        int oIdx2 = oIdx1 + out.strides[2];

        int iIdx0 = gx * in.strides[0] + gy * in.strides[1];
        int iIdx1 = iIdx0 + in.strides[2];
        int iIdx2 = iIdx1 + in.strides[2];

        if(isHSV2RGB) {
            T H = src[iIdx0];
            T S = src[iIdx1];
            T V = src[iIdx2];

            T R, G, B;
            R = G = B = 0;

            int   i = (int)(H * 6);
            T f = H * 6 - i;
            T p = V * (1 - S);
            T q = V * (1 - f * S);
            T t = V * (1 - (1 - f) * S);

            switch (i % 6) {
                case 0: R = V, G = t, B = p; break;
                case 1: R = q, G = V, B = p; break;
                case 2: R = p, G = V, B = t; break;
                case 3: R = p, G = q, B = V; break;
                case 4: R = t, G = p, B = V; break;
                case 5: R = V, G = p, B = q; break;
            }

            dst[oIdx0] = R;
            dst[oIdx1] = G;
            dst[oIdx2] = B;
        } else {
            T R = src[iIdx0];
            T G = src[iIdx1];
            T B = src[iIdx2];
            T Cmax = fmax(fmax(R, G), B);
            T Cmin = fmin(fmin(R, G), B);
            T delta= Cmax-Cmin;

            T H = 0;

            if (Cmax!=Cmin) {
                if (Cmax==R) H = (G-B)/delta + (G<B ? 6 : 0);
                if (Cmax==G) H = (B-R)/delta + 2;
                if (Cmax==B) H = (R-G)/delta + 4;
                H = H / 6.0f;
            }

            dst[oIdx0] = H;
            dst[oIdx1] = Cmax==0.0f ? 0 : delta/Cmax;
            dst[oIdx2] = Cmax;
        }
    }
}

template<typename T, bool isHSV2RGB>
void hsv2rgb_convert(Param<T> out, CParam<T> in)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    // all images are three channels, so batch
    // parameter would be along 4th dimension
    dim3 blocks(blk_x*in.dims[3], blk_y);

    CUDA_LAUNCH((convert<T, isHSV2RGB>), blocks, threads, out, in, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
