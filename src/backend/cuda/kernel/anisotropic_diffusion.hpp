/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include <ops.hpp>

namespace cuda {
namespace kernel {
static const int THREADS_X = 32;
static const int THREADS_Y = 8;

inline __device__ int clamp(const int value, const int low, const int high) {
    return max(low, min(value, high));
}

__forceinline__ __device__ int index(const int x, const int y, const int dim0,
                                     const int dim1, const int stride0,
                                     const int stride1) {
    return clamp(x, 0, dim0 - 1) * stride0 + clamp(y, 0, dim1 - 1) * stride1;
}

__device__ float quadratic(const float value) { return 1.0 / (1.0 + value); }

__device__ float computeGradientBasedUpdate(const float mct, const float C,
                                            const float S, const float N,
                                            const float W, const float E,
                                            const float SE, const float SW,
                                            const float NE, const float NW,
                                            const af_flux_function fftype) {
    float delta = 0;

    float dx, dy, df, db, cx, cxd;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df = E - C;
    db = C - W;

    if (fftype == AF_FLUX_EXPONENTIAL) {
        cx  = expf((df * df + 0.25f * powf(dy + 0.5f * (SE - NE), 2)) * mct);
        cxd = expf((db * db + 0.25f * powf(dy + 0.5f * (SW - NW), 2)) * mct);
    } else {
        cx =
            quadratic((df * df + 0.25f * powf(dy + 0.5f * (SE - NE), 2)) * mct);
        cxd =
            quadratic((db * db + 0.25f * powf(dy + 0.5f * (SW - NW), 2)) * mct);
    }
    delta += (cx * df - cxd * db);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    if (fftype == AF_FLUX_EXPONENTIAL) {
        cx  = expf((df * df + 0.25f * powf(dx + 0.5f * (SE - SW), 2)) * mct);
        cxd = expf((db * db + 0.25f * powf(dx + 0.5f * (NE - NW), 2)) * mct);
    } else {
        cx =
            quadratic((df * df + 0.25f * powf(dx + 0.5f * (SE - SW), 2)) * mct);
        cxd =
            quadratic((db * db + 0.25f * powf(dx + 0.5f * (NE - NW), 2)) * mct);
    }
    delta += (cx * df - cxd * db);

    return delta;
}

__device__ float computeCurvatureBasedUpdate(const float mct, const float C,
                                             const float S, const float N,
                                             const float W, const float E,
                                             const float SE, const float SW,
                                             const float NE, const float NW,
                                             const af_flux_function fftype) {
    float delta     = 0;
    float prop_grad = 0;

    float df0, db0;
    float dx, dy, df, db, cx, cxd, gmf, gmb, gmsqf, gmsqb;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df  = E - C;
    db  = C - W;
    df0 = df;
    db0 = db;

    gmsqf = (df * df + 0.25f * powf(dy + 0.5f * (SE - NE), 2));
    gmsqb = (db * db + 0.25f * powf(dy + 0.5f * (SW - NW), 2));

    gmf = sqrtf(1.0e-10 + gmsqf);
    gmb = sqrtf(1.0e-10 + gmsqb);

    cx  = expf(gmsqf * mct);
    cxd = expf(gmsqb * mct);

    delta += ((df / gmf) * cx - (db / gmb) * cxd);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    gmsqf = (df * df + 0.25f * powf(dx + 0.5f * (SE - SW), 2));
    gmsqb = (db * db + 0.25f * powf(dx + 0.5f * (NE - NW), 2));
    gmf   = sqrtf(1.0e-10 + gmsqf);
    gmb   = sqrtf(1.0e-10 + gmsqb);

    cx  = expf(gmsqf * mct);
    cxd = expf(gmsqb * mct);

    delta += ((df / gmf) * cx - (db / gmb) * cxd);

    if (delta > 0) {
        prop_grad +=
            (powf(fminf(db0, 0.0f), 2.0f) + powf(fmaxf(df0, 0.0f), 2.0f));
        prop_grad +=
            (powf(fminf(db, 0.0f), 2.0f) + powf(fmaxf(df, 0.0f), 2.0f));
    } else {
        prop_grad +=
            (powf(fmaxf(db0, 0.0f), 2.0f) + powf(fminf(df0, 0.0f), 2.0f));
        prop_grad +=
            (powf(fmaxf(db, 0.0f), 2.0f) + powf(fminf(df, 0.0f), 2.0f));
    }

    return sqrtf(prop_grad) * delta;
}

template <typename T, bool isMCDE>
static __global__ void diffUpdate(Param<T> inout, const float dt,
                                  const float mct,
                                  const af_flux_function fftype,
                                  const unsigned blkX, const unsigned blkY) {
    const unsigned RADIUS          = 1;
    const unsigned SHRD_MEM_WIDTH  = THREADS_X + 2 * RADIUS;  // Coloumns
    const unsigned SHRD_MEM_HEIGHT = THREADS_Y + 2 * RADIUS;  // Rows

    __shared__ float shrdMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int b2 = blockIdx.x / blkX;
    const int b3 = blockIdx.y / blkY;

    const int gx = blockDim.x * (blockIdx.x - b2 * blkX) + lx;
    const int gy = blockDim.y * (blockIdx.y - b3 * blkY) + ly;

    T* img = (T*)inout.ptr + (b3 * inout.strides[3] + b2 * inout.strides[2]);

#pragma unroll
    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT;
         b += blockDim.y, gy2 += blockDim.y) {
#pragma unroll
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH;
             a += blockDim.x, gx2 += blockDim.x) {
            int idx = index(gx2 - RADIUS, gy2 - RADIUS, inout.dims[0],
                            inout.dims[1], inout.strides[0], inout.strides[1]);
            shrdMem[b][a] = img[idx];
        }
    }

    __syncthreads();

    if (gx < inout.dims[0] && gy < inout.dims[1]) {
        int i       = lx + RADIUS;
        int j       = ly + RADIUS;
        float C     = shrdMem[j][i];
        float delta = 0;

        if (isMCDE) {
            delta = computeCurvatureBasedUpdate(
                mct, C, shrdMem[j][i + 1], shrdMem[j][i - 1], shrdMem[j - 1][i],
                shrdMem[j + 1][i], shrdMem[j + 1][i + 1], shrdMem[j - 1][i + 1],
                shrdMem[j + 1][i - 1], shrdMem[j - 1][i - 1], fftype);
        } else {
            delta = computeGradientBasedUpdate(
                mct, C, shrdMem[j][i + 1], shrdMem[j][i - 1], shrdMem[j - 1][i],
                shrdMem[j + 1][i], shrdMem[j + 1][i + 1], shrdMem[j - 1][i + 1],
                shrdMem[j + 1][i - 1], shrdMem[j - 1][i - 1], fftype);
        }

        img[gx * inout.strides[0] + gy * inout.strides[1]] =
            (T)(C + delta * dt);
    }
}

template <typename T, bool isMCDE>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct,
                          const af_flux_function fftype) {
    dim3 threads(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.dims[0], threads.x);
    int blkY = divup(inout.dims[1], threads.y);

    dim3 blocks(blkX * inout.dims[2], blkY * inout.dims[3], 1);

    const int maxBlkY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    const int blkZ = divup(blocks.y, maxBlkY);

    if (blkZ > 1) {
        blocks.y = maxBlkY;
        blocks.z = blkZ;
    }

    CUDA_LAUNCH((diffUpdate<T, isMCDE>), blocks, threads, inout, dt, mct,
                fftype, blkX, blkY);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
