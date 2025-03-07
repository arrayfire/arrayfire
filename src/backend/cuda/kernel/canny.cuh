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

// hasChanged is a variable in kernel space
// used to track the convergence of
// the breath first search algorithm
__device__ int hasChanged = 0;

namespace arrayfire {
namespace cuda {

__forceinline__ __device__ int lIdx(int x, int y, int stride0, int stride1) {
    return (x * stride0 + y * stride1);
}

template<typename T>
__global__ void nonMaxSuppression(Param<float> output, CParam<T> in,
                                  CParam<T> dx, CParam<T> dy, unsigned nBBS0,
                                  unsigned nBBS1) {
    const unsigned SHRD_MEM_WIDTH  = THREADS_X + 2;  // Coloumns
    const unsigned SHRD_MEM_HEIGHT = THREADS_Y + 2;  // Rows

    // Declared shared memory with 1 pixel border
    __shared__ T shrdMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    // local thread indices
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = blockIdx.x / nBBS0;
    const unsigned b3 = blockIdx.y / nBBS1;

    // global indices
    const int gx = blockDim.x * (blockIdx.x - b2 * nBBS0) + lx;
    const int gy = blockDim.y * (blockIdx.y - b3 * nBBS1) + ly;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    const T* mag = (const T*)in.ptr + (b2 * in.strides[2] + b3 * in.strides[3]);
    const T* dX = (const T*)dx.ptr + (b2 * dx.strides[2] + b3 * dx.strides[3]) +
                  dx.strides[1] + 1;
    const T* dY = (const T*)dy.ptr + (b2 * dy.strides[2] + b3 * dy.strides[3]) +
                  dy.strides[1] + 1;
    T* out = (float*)output.ptr +
             (b2 * output.strides[2] + b3 * output.strides[3]) +
             output.strides[1] + 1;

    // pull image to shared memory
#pragma unroll
    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT && gy2 < in.dims[1];
         b += blockDim.y, gy2 += blockDim.y)
#pragma unroll
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH && gx2 < in.dims[0];
             a += blockDim.x, gx2 += blockDim.x)
            shrdMem[b][a] = mag[lIdx(gx2, gy2, in.strides[0], in.strides[1])];

    int i = lx + 1;
    int j = ly + 1;

    __syncthreads();

    if (gx < in.dims[0] - 2 && gy < in.dims[1] - 2) {
        int idx = lIdx(gx, gy, in.strides[0], in.strides[1]);

        const float cmag = shrdMem[j][i];

        if (cmag == 0.0f)
            out[idx] = (T)0;
        else {
            const float dx = dX[idx];
            const float dy = dY[idx];
            const float se = shrdMem[j + 1][i + 1];
            const float nw = shrdMem[j - 1][i - 1];
            const float ea = shrdMem[j][i + 1];
            const float we = shrdMem[j][i - 1];
            const float ne = shrdMem[j - 1][i + 1];
            const float sw = shrdMem[j + 1][i - 1];
            const float no = shrdMem[j - 1][i];
            const float so = shrdMem[j + 1][i];

            float a1, a2, b1, b2, alpha;

            if (dx >= 0) {
                if (dy >= 0) {
                    const bool isDxMagGreater = (dx - dy) >= 0;

                    a1    = isDxMagGreater ? ea : so;
                    a2    = isDxMagGreater ? we : no;
                    b1    = se;
                    b2    = nw;
                    alpha = isDxMagGreater ? dy / dx : dx / dy;
                } else {
                    const bool isDyMagGreater = (dx + dy) >= 0;

                    a1    = isDyMagGreater ? ea : no;
                    a2    = isDyMagGreater ? we : so;
                    b1    = ne;
                    b2    = sw;
                    alpha = isDyMagGreater ? -dy / dx : dx / -dy;
                }
            } else {
                if (dy >= 0) {
                    const bool isDxMagGreater = (dx + dy) >= 0;

                    a1    = isDxMagGreater ? so : we;
                    a2    = isDxMagGreater ? no : ea;
                    b1    = sw;
                    b2    = ne;
                    alpha = isDxMagGreater ? -dx / dy : dy / -dx;
                } else {
                    const bool isDyMagGreater = (-dx + dy) >= 0;

                    a1    = isDyMagGreater ? we : no;
                    a2    = isDyMagGreater ? ea : so;
                    b1    = nw;
                    b2    = se;
                    alpha = isDyMagGreater ? dy / dx : dx / dy;
                }
            }

            float mag1 = (1 - alpha) * a1 + alpha * b1;
            float mag2 = (1 - alpha) * a2 + alpha * b2;

            if (cmag > mag1 && cmag > mag2) {
                out[idx] = cmag;
            } else {
                out[idx] = (T)0;
            }
        }
    }
}

template<typename T>
__global__ void initEdgeOut(Param<T> output, CParam<T> strong, CParam<T> weak,
                            unsigned nBBS0, unsigned nBBS1) {
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = blockIdx.x / nBBS0;
    const unsigned b3 = blockIdx.y / nBBS1;

    // global indices
    const int gx = blockDim.x * (blockIdx.x - b2 * nBBS0) + threadIdx.x;
    const int gy = blockDim.y * (blockIdx.y - b3 * nBBS1) + threadIdx.y;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    const T* wPtr = weak.ptr + (b2 * weak.strides[2] + b3 * weak.strides[3]) +
                    weak.strides[1] + 1;
    const T* sPtr = strong.ptr +
                    (b2 * strong.strides[2] + b3 * strong.strides[3]) +
                    strong.strides[1] + 1;
    T* oPtr = output.ptr + (b2 * output.strides[2] + b3 * output.strides[3]) +
              output.strides[1] + 1;

    if (gx < (output.dims[0] - 2) && gy < (output.dims[1] - 2)) {
        int idx   = lIdx(gx, gy, output.strides[0], output.strides[1]);
        oPtr[idx] = (sPtr[idx] > 0 ? STRONG : (wPtr[idx] > 0 ? WEAK : NOEDGE));
    }
}

#define VALID_BLOCK_IDX(j, i)                             \
    ((j) > 0 && (j) < (SHRD_MEM_HEIGHT - 1) && (i) > 0 && \
     (i) < (SHRD_MEM_WIDTH - 1))

template<typename T>
__global__ void edgeTrack(Param<T> output, unsigned nBBS0, unsigned nBBS1) {
    const unsigned SHRD_MEM_WIDTH  = THREADS_X + 2;  // Cols
    const unsigned SHRD_MEM_HEIGHT = THREADS_Y + 2;  // Rows

    // shared memory with 1 pixel border
    // strong and weak images are binary(char) images thus,
    // occupying only (16+2)*(16+2) = 324 bytes per shared memory tile
    __shared__ int outMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    // local thread indices
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = blockIdx.x / nBBS0;
    const unsigned b3 = blockIdx.y / nBBS1;

    // global indices
    const int gx = blockDim.x * (blockIdx.x - b2 * nBBS0) + lx;
    const int gy = blockDim.y * (blockIdx.y - b3 * nBBS1) + ly;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    T* oPtr = output.ptr + (b2 * output.strides[2] + b3 * output.strides[3]);

    // pull image to shared memory
#pragma unroll
    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT;
         b += blockDim.y, gy2 += blockDim.y) {
#pragma unroll
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH;
             a += blockDim.x, gx2 += blockDim.x) {
            int x = gx2;
            int y = gy2;
            if (x >= 0 && x < output.dims[0] && y >= 0 && y < output.dims[1])
                outMem[b][a] =
                    oPtr[lIdx(x, y, output.strides[0], output.strides[1])];
            else
                outMem[b][a] = NOEDGE;
        }
    }

    int i = lx + 1;
    int j = ly + 1;

    __syncthreads();

    int continueIter = 1;

    while (continueIter) {
        int nw, no, ne, we, ea, sw, so, se;

        if (outMem[j][i] == WEAK) {
            nw = outMem[j - 1][i - 1];
            no = outMem[j - 1][i];
            ne = outMem[j - 1][i + 1];
            we = outMem[j][i - 1];
            ea = outMem[j][i + 1];
            sw = outMem[j + 1][i - 1];
            so = outMem[j + 1][i];
            se = outMem[j + 1][i + 1];

            bool hasStrongNeighbour =
                nw == STRONG || no == STRONG || ne == STRONG || ea == STRONG ||
                se == STRONG || so == STRONG || sw == STRONG || we == STRONG;

            if (hasStrongNeighbour) outMem[j][i] = STRONG;
        }

        __syncthreads();

        // Check if there are any STRONG pixels with weak neighbours.
        // This search however ignores 1-pixel border encompassing the
        // shared memory tile region.
        bool hasWeakNeighbour = false;
        if (outMem[j][i] == STRONG) {
            nw = outMem[j - 1][i - 1] == WEAK && VALID_BLOCK_IDX(j - 1, i - 1);
            no = outMem[j - 1][i] == WEAK && VALID_BLOCK_IDX(j - 1, i);
            ne = outMem[j - 1][i + 1] == WEAK && VALID_BLOCK_IDX(j - 1, i + 1);
            we = outMem[j][i - 1] == WEAK && VALID_BLOCK_IDX(j, i - 1);
            ea = outMem[j][i + 1] == WEAK && VALID_BLOCK_IDX(j, i + 1);
            sw = outMem[j + 1][i - 1] == WEAK && VALID_BLOCK_IDX(j + 1, i - 1);
            so = outMem[j + 1][i] == WEAK && VALID_BLOCK_IDX(j + 1, i);
            se = outMem[j + 1][i + 1] == WEAK && VALID_BLOCK_IDX(j + 1, i + 1);

            hasWeakNeighbour = nw || no || ne || ea || se || so || sw || we;
        }

        continueIter = __syncthreads_or(hasWeakNeighbour);
    };

    // Check if any 1-pixel border ring
    // has weak pixels with strong candidates
    // within the main region, then increment hasChanged.
    int cu = outMem[j][i];
    int nw = outMem[j - 1][i - 1];
    int no = outMem[j - 1][i];
    int ne = outMem[j - 1][i + 1];
    int ea = outMem[j][i + 1];
    int se = outMem[j + 1][i + 1];
    int so = outMem[j + 1][i];
    int sw = outMem[j + 1][i - 1];
    int we = outMem[j][i - 1];

    bool hasWeakNeighbour = nw == WEAK || no == WEAK || ne == WEAK ||
                            ea == WEAK || se == WEAK || so == WEAK ||
                            sw == WEAK || we == WEAK;

    if (__syncthreads_or(cu == STRONG && hasWeakNeighbour) && lx == 0 &&
        ly == 0)
        atomicAdd(&hasChanged, 1);

    // Update output with shared memory result
    if (gx < (output.dims[0] - 2) && gy < (output.dims[1] - 2))
        oPtr[lIdx(gx, gy, output.strides[0], output.strides[1]) +
             output.strides[1] + 1] = outMem[j][i];
}

template<typename T>
__global__ void suppressLeftOver(Param<T> output, unsigned nBBS0,
                                 unsigned nBBS1) {
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = blockIdx.x / nBBS0;
    const unsigned b3 = blockIdx.y / nBBS1;

    // global indices
    const int gx = blockDim.x * (blockIdx.x - b2 * nBBS0) + threadIdx.x;
    const int gy = blockDim.y * (blockIdx.y - b3 * nBBS1) + threadIdx.y;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    T* oPtr = output.ptr + (b2 * output.strides[2] + b3 * output.strides[3]) +
              output.strides[1] + 1;

    if (gx < (output.dims[0] - 2) && gy < (output.dims[1] - 2)) {
        int idx = lIdx(gx, gy, output.strides[0], output.strides[1]);
        T val   = oPtr[idx];
        if (val == WEAK) oPtr[idx] = NOEDGE;
    }
}

}  // namespace cuda
}  // namespace arrayfire
