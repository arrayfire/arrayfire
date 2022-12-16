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
#include <shared.hpp>

__constant__ char cFilter[2 * (2 * (MAX_CONV1_FILTER_LEN - 1) + CONV_THREADS) *
                          sizeof(double)];

namespace arrayfire {
namespace cuda {

template<typename T, typename aT, bool expand>
__global__ void convolve1(Param<T> out, CParam<T> signal, int fLen, int nBBS0,
                          int nBBS1, int o1, int o2, int o3, int s1, int s2,
                          int s3) {
    SharedMemory<T> shared;
    T *shrdMem = shared.getPointer();

    const int padding = fLen - 1;
    const int shrdLen = blockDim.x + 2 * padding;
    const unsigned b1 = blockIdx.x / nBBS0; /* [0 {1} 2 3] */
    const unsigned b3 =
        (blockIdx.y + blockIdx.z * gridDim.y) / nBBS1; /* [0 1 2 {3}] */
    const unsigned b2 =
        (blockIdx.y + blockIdx.z * gridDim.y) - nBBS1 * b3; /* [0 1 {2} 3] */
    if (b2 >= out.dims[2] || b3 >= out.dims[3]) return;

    T *dst = (T *)out.ptr +
             (b1 * out.strides[1] + /* activated with batched input signal */
              o1 * out.strides[1] + /* activated with batched input filter */
              b2 * out.strides[2] + /* activated with batched input signal */
              o2 * out.strides[2] + /* activated with batched input filter */
              b3 * out.strides[3] + /* activated with batched input signal */
              o3 * out.strides[3]); /* activated with batched input filter */

    const T *src =
        (const T *)signal.ptr +
        (b1 * signal.strides[1] + /* activated with batched input signal */
         s1 * signal.strides[1] + /* activated with batched input filter */
         b2 * signal.strides[2] + /* activated with batched input signal */
         s2 * signal.strides[2] + /* activated with batched input filter */
         b3 * signal.strides[3] + /* activated with batched input signal */
         s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse = (const aT *)cFilter;

    int gx = blockDim.x * (blockIdx.x - b1 * nBBS0);

    int s0 = signal.strides[0];
    int d0 = signal.dims[0];
    for (int i = threadIdx.x; i < shrdLen; i += blockDim.x) {
        int idx    = gx - padding + i;
        shrdMem[i] = (idx >= 0 && idx < d0) ? src[idx * s0] : scalar<T>(0);
    }
    __syncthreads();
    gx += threadIdx.x;

    if (gx < out.dims[0]) {
        int lx   = threadIdx.x + padding + (expand ? 0 : fLen >> 1);
        aT accum = scalar<aT>(0);
        for (int f = 0; f < fLen; ++f) {
            accum = accum + (shrdMem[lx - f] * impulse[f]);
        }
        dst[gx] = (T)accum;
    }
}

}  // namespace cuda
}  // namespace arrayfire
