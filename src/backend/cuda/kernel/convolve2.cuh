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

__constant__ char cFilter[2 * (2 * (MAX_CONV1_FILTER_LEN - 1) + CONV_THREADS) *
                          sizeof(double)];

namespace arrayfire {
namespace cuda {

template<typename T, typename aT, bool expand, int fLen0, int fLen1>
__global__ void convolve2(Param<T> out, CParam<T> signal, int nBBS0, int nBBS1,
                          int o2, int o3, int s2, int s3) {
    const size_t C_SIZE = (CONV2_THREADS_X + 2 * (fLen0 - 1)) *
                          (CONV2_THREADS_Y + 2 * (fLen1 - 1));
    __shared__ T shrdMem[C_SIZE];

    const int radius0  = fLen0 - 1;
    const int radius1  = fLen1 - 1;
    const int padding0 = 2 * radius0;
    const int padding1 = 2 * radius1;
    const int shrdLen0 = CONV2_THREADS_X + padding0;
    const int shrdLen1 = CONV2_THREADS_Y + padding1;

    unsigned b0 = blockIdx.x / nBBS0;
    unsigned b1 = (blockIdx.y + blockIdx.z * gridDim.y) / nBBS1;
    T *dst      = (T *)out.ptr +
             (b0 * out.strides[2] + /* activated with batched input signal */
              o2 * out.strides[2] + /* activated with batched input filter */
              b1 * out.strides[3] + /* activated with batched input signal */
              o3 * out.strides[3]); /* activated with batched input filter */

    const T *src =
        (const T *)signal.ptr +
        (b0 * signal.strides[2] + /* activated with batched input signal */
         s2 * signal.strides[2] + /* activated with batched input filter */
         b1 * signal.strides[3] + /* activated with batched input signal */
         s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse = (const aT *)cFilter;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int gx = CONV2_THREADS_X * (blockIdx.x - b0 * nBBS0) + lx;
    int gy =
        CONV2_THREADS_Y * ((blockIdx.y + blockIdx.z * gridDim.y) - b1 * nBBS1) +
        ly;

    if (b1 >= out.dims[3]) return;

    int s0 = signal.strides[0];
    int s1 = signal.strides[1];
    int d0 = signal.dims[0];
    int d1 = signal.dims[1];
    // below loops are traditional loops, they only run multiple
    // times filter length is more than launch size
#pragma unroll
    for (int b = ly, gy2 = gy; b < shrdLen1;
         b += CONV2_THREADS_Y, gy2 += CONV2_THREADS_Y) {
        int j     = gy2 - radius1;
        bool is_j = j >= 0 && j < d1;
        // move row_set CONV2_THREADS_Y along coloumns
#pragma unroll
        for (int a = lx, gx2 = gx; a < shrdLen0;
             a += CONV2_THREADS_X, gx2 += CONV2_THREADS_X) {
            int i     = gx2 - radius0;
            bool is_i = i >= 0 && i < d0;
            shrdMem[b * shrdLen0 + a] =
                (is_i && is_j ? src[i * s0 + j * s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (gx < out.dims[0] && gy < out.dims[1]) {
        int ci = lx + radius0 + (expand ? 0 : fLen0 >> 1);
        int cj = ly + radius1 + (expand ? 0 : fLen1 >> 1);

        aT accum = scalar<aT>(0);
#pragma unroll
        for (int fj = 0; fj < fLen1; ++fj) {
#pragma unroll
            for (int fi = 0; fi < fLen0; ++fi) {
                aT f_val = impulse[fj * fLen0 + fi];
                T s_val  = shrdMem[(cj - fj) * shrdLen0 + (ci - fi)];
                accum    = accum + s_val * f_val;
            }
        }
        dst[gy * out.strides[1] + gx] = (T)accum;
    }
}

}  // namespace cuda
}  // namespace arrayfire
