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

__inline__ int index(int i, int j, int k, int jstride, int kstride) {
    return i + j * jstride + k * kstride;
}

template<typename T, typename aT, bool expand>
__global__ void convolve3(Param<T> out, CParam<T> signal, int fLen0, int fLen1,
                          int fLen2, int nBBS, int o3, int s3) {
    SharedMemory<T> shared;

    T *shrdMem   = shared.getPointer();
    int radius0  = fLen0 - 1;
    int radius1  = fLen1 - 1;
    int radius2  = fLen2 - 1;
    int shrdLen0 = blockDim.x + 2 * radius0;
    int shrdLen1 = blockDim.y + 2 * radius1;
    int shrdLen2 = blockDim.z + 2 * radius2;
    int skStride = shrdLen0 * shrdLen1;
    int fStride  = fLen0 * fLen1;
    unsigned b2  = blockIdx.x / nBBS;

    T *dst = (T *)out.ptr +
             (b2 * out.strides[3] + /* activated with batched input signal */
              o3 * out.strides[3]); /* activated with batched input filter */

    const T *src =
        (const T *)signal.ptr +
        (b2 * signal.strides[3] + /* activated with batched input signal */
         s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse = (const aT *)cFilter;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lz = threadIdx.z;
    int gx = blockDim.x * (blockIdx.x - b2 * nBBS) + lx;
    int gy = blockDim.y * blockIdx.y + ly;
    int gz = blockDim.z * blockIdx.z + lz;

    int s0 = signal.strides[0];
    int s1 = signal.strides[1];
    int s2 = signal.strides[2];
    int d0 = signal.dims[0];
    int d1 = signal.dims[1];
    int d2 = signal.dims[2];
#pragma unroll
    for (int c = lz, gz2 = gz; c < shrdLen2;
         c += CONV3_CUBE_Z, gz2 += CONV3_CUBE_Z) {
        int k     = gz2 - radius2;
        bool is_k = k >= 0 && k < d2;
#pragma unroll
        for (int b = ly, gy2 = gy; b < shrdLen1;
             b += CONV3_CUBE_Y, gy2 += CONV3_CUBE_Y) {
            int j     = gy2 - radius1;
            bool is_j = j >= 0 && j < d1;
#pragma unroll
            for (int a = lx, gx2 = gx; a < shrdLen0;
                 a += CONV3_CUBE_X, gx2 += CONV3_CUBE_X) {
                int i     = gx2 - radius0;
                bool is_i = i >= 0 && i < d0;
                shrdMem[c * skStride + b * shrdLen0 + a] =
                    (is_i && is_j && is_k ? src[i * s0 + j * s1 + k * s2]
                                          : scalar<T>(0));
            }
        }
    }
    __syncthreads();

    if (gx < out.dims[0] && gy < out.dims[1] && gz < out.dims[2]) {
        int ci = lx + radius0 + (expand ? 0 : fLen0 >> 1);
        int cj = ly + radius1 + (expand ? 0 : fLen1 >> 1);
        int ck = lz + radius2 + (expand ? 0 : fLen2 >> 1);

        aT accum = scalar<aT>(0);
#pragma unroll
        for (int fk = 0; fk < fLen2; ++fk) {
#pragma unroll
            for (int fj = 0; fj < fLen1; ++fj) {
#pragma unroll
                for (int fi = 0; fi < fLen0; ++fi) {
                    aT f_val = impulse[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = shrdMem[index(ci - fi, cj - fj, ck - fk, shrdLen0,
                                            skStride)];
                    accum   = accum + s_val * f_val;
                }
            }
        }
        dst[index(gx, gy, gz, out.strides[1], out.strides[2])] = (T)accum;
    }
}

}  // namespace cuda
}  // namespace arrayfire
