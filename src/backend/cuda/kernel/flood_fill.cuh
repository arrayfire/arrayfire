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
#include <af/defines.h>

/// doAnotherLaunch is a variable in kernel space
/// used to track the convergence of
/// the breath first search algorithm
__device__ int doAnotherLaunch = 0;

namespace arrayfire {
namespace cuda {

/// Output array is set to the following values during the progression
/// of the algorithm.
///
/// 0 - not processed
/// 1 - not valid
/// 2 - valid (candidate for neighborhood walk, pushed onto the queue)
///
/// Once, the algorithm is finished, output is reset
/// to either zero or \p newValue for all valid pixels.
template<typename T>
constexpr T VALID() {
    return T(2);
}
template<typename T>
constexpr T INVALID() {
    return T(1);
}
template<typename T>
constexpr T ZERO() {
    return T(0);
}

template<typename T>
__global__ void initSeeds(Param<T> out, CParam<uint> seedsx,
                          CParam<uint> seedsy) {
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < seedsx.elements()) {
        uint x                       = seedsx.ptr[idx];
        uint y                       = seedsy.ptr[idx];
        out.ptr[x + y * out.dims[0]] = VALID<T>();
    }
}

template<typename T>
__global__ void floodStep(Param<T> out, CParam<T> img, T lowValue,
                          T highValue) {
    constexpr int RADIUS      = 1;
    constexpr int SMEM_WIDTH  = THREADS_X + 2 * RADIUS;
    constexpr int SMEM_HEIGHT = THREADS_Y + 2 * RADIUS;

    __shared__ T smem[SMEM_HEIGHT][SMEM_WIDTH];

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int gx = blockDim.x * blockIdx.x + lx;
    const int gy = blockDim.y * blockIdx.y + ly;
    const int d0 = out.dims[0];
    const int d1 = out.dims[1];
    const int s0 = out.strides[0];
    const int s1 = out.strides[1];

    const T *iptr = (const T *)img.ptr;
    T *optr       = (T *)out.ptr;
#pragma unroll
    for (int b = ly, gy2 = gy; b < SMEM_HEIGHT;
         b += blockDim.y, gy2 += blockDim.y) {
#pragma unroll
        for (int a = lx, gx2 = gx; a < SMEM_WIDTH;
             a += blockDim.x, gx2 += blockDim.x) {
            int x      = gx2 - RADIUS;
            int y      = gy2 - RADIUS;
            bool inROI = (x >= 0 && x < d0 && y >= 0 && y < d1);
            smem[b][a] = (inROI ? optr[x * s0 + y * s1] : INVALID<T>());
        }
    }
    int i = lx + RADIUS;
    int j = ly + RADIUS;

    T tImgVal = iptr[(clamp(gx, 0, int(img.dims[0] - 1)) * img.strides[0] +
                      clamp(gy, 0, int(img.dims[1] - 1)) * img.strides[1])];
    const int isPxBtwnThresholds =
        (tImgVal >= lowValue && tImgVal <= highValue);
    __syncthreads();

    T origOutVal      = smem[j][i];
    bool blockChanged = false;
    bool isBorderPxl  = (lx == 0 || ly == 0 || lx == (blockDim.x - 1) ||
                        ly == (blockDim.y - 1));
    do {
        int validNeighbors = 0;
#pragma unroll
        for (int no_j = -RADIUS; no_j <= RADIUS; ++no_j) {
#pragma unroll
            for (int no_i = -RADIUS; no_i <= RADIUS; ++no_i) {
                T currVal = smem[j + no_j][i + no_i];
                validNeighbors += (currVal == VALID<T>());
            }
        }
        __syncthreads();

        bool outChanged = (smem[j][i] == ZERO<T>() && (validNeighbors > 0));
        if (outChanged) { smem[j][i] = T(isPxBtwnThresholds + INVALID<T>()); }
        blockChanged = __syncthreads_or(int(outChanged));
    } while (blockChanged);

    T newOutVal = smem[j][i];

    bool borderChanged =
        (isBorderPxl && newOutVal != origOutVal && newOutVal == VALID<T>());

    borderChanged = __syncthreads_or(int(borderChanged));

    if (borderChanged && lx == 0 && ly == 0) {
        // Atleast one border pixel changed. Therefore, mark for
        // another kernel launch to propogate changes beyond border
        // of this block
        doAnotherLaunch = 1;
    }

    if (gx < d0 && gy < d1) { optr[(gx * s0 + gy * s1)] = smem[j][i]; }
}

template<typename T>
__global__ void finalizeOutput(Param<T> out, T newValue) {
    uint gx = blockDim.x * blockIdx.x + threadIdx.x;
    uint gy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gx < out.dims[0] && gy < out.dims[1]) {
        uint idx     = gx * out.strides[0] + gy * out.strides[1];
        T val        = out.ptr[idx];
        out.ptr[idx] = (val == VALID<T>() ? newValue : ZERO<T>());
    }
}

}  // namespace cuda
}  // namespace arrayfire
