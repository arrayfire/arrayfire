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

namespace arrayfire {
namespace cuda {

__device__ int reflect101(int index, int endIndex) {
    return abs(endIndex - abs(endIndex - index));
}

template<typename Ti>
__device__ Ti load2ShrdMem(const Ti* in, int d0, int d1, int gx, int gy,
                           int inStride1, int inStride0) {
    int idx =
        reflect101(gx, d0 - 1) * inStride0 + reflect101(gy, d1 - 1) * inStride1;
    return in[idx];
}

template<typename Ti, typename To>
__global__ void sobel3x3(Param<To> dx, Param<To> dy, CParam<Ti> in, int nBBS0,
                         int nBBS1) {
    __shared__ Ti shrdMem[THREADS_X + 2][THREADS_Y + 2];

    // calculate necessary offset and window parameters
    const int radius  = 1;
    const int padding = 2 * radius;
    const int shrdLen = blockDim.x + padding;

    // batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const Ti* iptr =
        (const Ti*)in.ptr + (b2 * in.strides[2] + b3 * in.strides[3]);
    To* dxptr = (To*)dx.ptr + (b2 * dx.strides[2] + b3 * dx.strides[3]);
    To* dyptr = (To*)dy.ptr + (b2 * dy.strides[2] + b3 * dy.strides[3]);

    // local neighborhood indices
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // global indices
    int gx = THREADS_X * (blockIdx.x - b2 * nBBS0) + lx;
    int gy = THREADS_Y * (blockIdx.y - b3 * nBBS1) + ly;

    for (int b = ly, gy2 = gy; b < shrdLen;
         b += blockDim.y, gy2 += blockDim.y) {
        for (int a = lx, gx2 = gx; a < shrdLen;
             a += blockDim.x, gx2 += blockDim.x) {
            shrdMem[a][b] =
                load2ShrdMem<Ti>(iptr, in.dims[0], in.dims[1], gx2 - radius,
                                 gy2 - radius, in.strides[1], in.strides[0]);
        }
    }

    __syncthreads();

    // Only continue if we're at a valid location
    if (gx < in.dims[0] && gy < in.dims[1]) {
        int i  = lx + radius;
        int j  = ly + radius;
        int _i = i - 1;
        int i_ = i + 1;
        int _j = j - 1;
        int j_ = j + 1;

        float NW = shrdMem[_i][_j];
        float SW = shrdMem[i_][_j];
        float NE = shrdMem[_i][j_];
        float SE = shrdMem[i_][j_];

        float t1                       = shrdMem[_i][j];
        float t2                       = shrdMem[i_][j];
        dxptr[gy * dx.strides[1] + gx] = (SW + SE - (NW + NE) + 2 * (t2 - t1));

        t1                             = shrdMem[i][_j];
        t2                             = shrdMem[i][j_];
        dyptr[gy * dy.strides[1] + gx] = (NE + SE - (NW + SW) + 2 * (t2 - t1));
    }
}

}  // namespace cuda
}  // namespace arrayfire
