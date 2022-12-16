/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <math.hpp>

__constant__ char sFilter[2 * SCONV_THREADS_Y *
                          (2 * (MAX_SCONV_FILTER_LEN - 1) + SCONV_THREADS_X) *
                          sizeof(double)];

namespace arrayfire {
namespace cuda {

template<typename T, typename accType, int conv_dim, bool expand, int fLen>
__global__ void convolve2_separable(Param<T> out, CParam<T> signal, int nBBS0,
                                    int nBBS1) {
    const int smem_len =
        (conv_dim == 0 ? (SCONV_THREADS_X + 2 * (fLen - 1)) * SCONV_THREADS_Y
                       : (SCONV_THREADS_Y + 2 * (fLen - 1)) * SCONV_THREADS_X);
    __shared__ T shrdMem[smem_len];

    const int radius  = fLen - 1;
    const int padding = 2 * radius;
    const int s0      = signal.strides[0];
    const int s1      = signal.strides[1];
    const int d0      = signal.dims[0];
    const int d1      = signal.dims[1];
    const int shrdLen = SCONV_THREADS_X + (conv_dim == 0 ? padding : 0);

    unsigned b2  = blockIdx.x / nBBS0;
    unsigned b3  = blockIdx.y / nBBS1;
    T *dst       = (T *)out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);
    const T *src = (const T *)signal.ptr +
                   (b2 * signal.strides[2] + b3 * signal.strides[3]);
    const accType *impulse = (const accType *)sFilter;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int ox = SCONV_THREADS_X * (blockIdx.x - b2 * nBBS0) + lx;
    int oy = SCONV_THREADS_Y * (blockIdx.y - b3 * nBBS1) + ly;
    int gx = ox;
    int gy = oy;

    // below if-else statement is based on template parameter
    if (conv_dim == 0) {
        gx += (expand ? 0 : fLen >> 1);
        int endX = ((fLen - 1) << 1) + SCONV_THREADS_X;

#pragma unroll
        for (int lx = threadIdx.x, glb_x = gx; lx < endX;
             lx += SCONV_THREADS_X, glb_x += SCONV_THREADS_X) {
            int i     = glb_x - radius;
            int j     = gy;
            bool is_i = i >= 0 && i < d0;
            bool is_j = j >= 0 && j < d1;
            shrdMem[ly * shrdLen + lx] =
                (is_i && is_j ? src[i * s0 + j * s1] : scalar<T>(0));
        }

    } else if (conv_dim == 1) {
        gy += (expand ? 0 : fLen >> 1);
        int endY = ((fLen - 1) << 1) + SCONV_THREADS_Y;

#pragma unroll
        for (int ly = threadIdx.y, glb_y = gy; ly < endY;
             ly += SCONV_THREADS_Y, glb_y += SCONV_THREADS_Y) {
            int i     = gx;
            int j     = glb_y - radius;
            bool is_i = i >= 0 && i < d0;
            bool is_j = j >= 0 && j < d1;
            shrdMem[ly * shrdLen + lx] =
                (is_i && is_j ? src[i * s0 + j * s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (ox < out.dims[0] && oy < out.dims[1]) {
        // below conditional statement is based on template parameter
        int i         = (conv_dim == 0 ? lx : ly) + radius;
        accType accum = scalar<accType>(0);
#pragma unroll
        for (int f = 0; f < fLen; ++f) {
            accType f_val = impulse[f];
            // below conditional statement is based on template parameter
            int s_idx = (conv_dim == 0 ? (ly * shrdLen + (i - f))
                                       : ((i - f) * shrdLen + lx));
            T s_val   = shrdMem[s_idx];
            accum     = accum + s_val * f_val;
        }
        dst[oy * out.strides[1] + ox] = (T)accum;
    }
}

}  // namespace cuda
}  // namespace arrayfire
