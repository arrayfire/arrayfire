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

namespace arrayfire {
namespace cuda {

inline __device__ int lIdx(int x, int y, int stride1, int stride0) {
    return (y * stride1 + x * stride0);
}

template<typename inType, typename outType>
inline __device__ void load2ShrdMem(outType *shrd, const inType *const in,
                                    int lx, int ly, int shrdStride, int dim0,
                                    int dim1, int gx, int gy, int inStride1,
                                    int inStride0) {
    shrd[ly * shrdStride + lx] = in[lIdx(
        clamp(gx, 0, dim0 - 1), clamp(gy, 0, dim1 - 1), inStride1, inStride0)];
}

template<typename inType, typename outType>
__global__ void bilateral(Param<outType> out, CParam<inType> in,
                          float sigma_space, float sigma_color, int gaussOff,
                          int nBBS0, int nBBS1) {
    SharedMemory<outType> shared;
    outType *localMem = shared.getPointer();
    outType *gauss2d  = localMem + gaussOff;

    const int radius                    = max((int)(sigma_space * 1.5f), 1);
    const int padding                   = 2 * radius;
    const int window_size               = padding + 1;
    const int shrdLen                   = THREADS_X + padding;
    const float variance_range          = sigma_color * sigma_color;
    const float variance_space          = sigma_space * sigma_space;
    const float variance_space_neg2     = -2.0 * variance_space;
    const float inv_variance_range_neg2 = -0.5 / variance_range;

    // gfor batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const inType *iptr =
        (const inType *)in.ptr + (b2 * in.strides[2] + b3 * in.strides[3]);
    outType *optr =
        (outType *)out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    const int gx = THREADS_X * (blockIdx.x - b2 * nBBS0) + lx;
    const int gy = THREADS_Y * (blockIdx.y - b3 * nBBS1) + ly;

    // generate gauss2d spatial variance values for block
    if (lx < window_size && ly < window_size) {
        int x = lx - radius;
        int y = ly - radius;
        gauss2d[ly * window_size + lx] =
            __expf(((x * x) + (y * y)) / variance_space_neg2);
    }

    // pull image to local memory
    for (int b = ly, gy2 = gy; b < shrdLen;
         b += blockDim.y, gy2 += blockDim.y) {
        // move row_set get_local_size(1) along coloumns
        for (int a = lx, gx2 = gx; a < shrdLen;
             a += blockDim.x, gx2 += blockDim.x) {
            load2ShrdMem<inType, outType>(
                localMem, iptr, a, b, shrdLen, in.dims[0], in.dims[1],
                gx2 - radius, gy2 - radius, in.strides[1], in.strides[0]);
        }
    }

    __syncthreads();

    if (gx < in.dims[0] && gy < in.dims[1]) {
        lx += radius;
        ly += radius;
        const outType center_color = localMem[ly * shrdLen + lx];
        outType res                = 0;
        outType norm               = 0;
        int joff                   = (ly - radius) * shrdLen + (lx - radius);
        int goff                   = 0;

#pragma unroll
        for (int wj = 0; wj < window_size; ++wj) {
#pragma unroll
            for (int wi = 0; wi < window_size; ++wi) {
                const outType tmp_color = localMem[joff + wi];
                const outType c         = center_color - tmp_color;
                const outType gauss_range =
                    __expf(c * c * inv_variance_range_neg2);
                const outType weight = gauss2d[goff + wi] * gauss_range;
                norm += weight;
                res += tmp_color * weight;
            }
            joff += shrdLen;
            goff += window_size;
        }
        optr[gy * out.strides[1] + gx] = res / norm;
    }
}

}  // namespace cuda
}  // namespace arrayfire
