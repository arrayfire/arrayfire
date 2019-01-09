/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__constant int STRONG = 1;
__constant int WEAK   = 2;
__constant int NOEDGE = 0;

#if defined(INIT_EDGE_OUT)
__kernel void initEdgeOutKernel(__global T* output, KParam oInfo,
                                __global const T* strong, KParam sInfo,
                                __global const T* weak, KParam wInfo,
                                unsigned nBBS0, unsigned nBBS1) {
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx =
        get_local_size(0) * (get_group_id(0) - b2 * nBBS0) + get_local_id(0);
    const int gy =
        get_local_size(1) * (get_group_id(1) - b3 * nBBS1) + get_local_id(1);

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global const T* wPtr =
        weak + (b2 * wInfo.strides[2] + b3 * wInfo.strides[3] + wInfo.offset) +
        wInfo.strides[1] + 1;

    __global const T* sPtr = strong + (b2 * sInfo.strides[2] +
                                       b3 * sInfo.strides[3] + sInfo.offset) +
                             sInfo.strides[1] + 1;

    __global T* oPtr = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3] +
                                 oInfo.offset) +
                       oInfo.strides[1] + 1;

    if (gx < (oInfo.dims[0] - 2) && gy < (oInfo.dims[1] - 2)) {
        int idx   = gx * oInfo.strides[0] + gy * oInfo.strides[1];
        oPtr[idx] = (sPtr[idx] > 0 ? STRONG : (wPtr[idx] > 0 ? WEAK : NOEDGE));
    }
}
#endif

#define VALID_BLOCK_IDX(j, i)                             \
    ((j) > 0 && (j) < (SHRD_MEM_HEIGHT - 1) && (i) > 0 && \
     (i) < (SHRD_MEM_WIDTH - 1))

#if defined(EDGE_TRACER)
__kernel void edgeTrackKernel(__global T* output, KParam oInfo, unsigned nBBS0,
                              unsigned nBBS1,
                              __global volatile int* hasChanged) {
    // shared memory with 1 pixel border
    // strong and weak images are binary(char) images thus,
    // occupying only (16+2)*(16+2) = 324 bytes per shared memory tile
    __local int outMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];
    __local int predicates[TOTAL_NUM_THREADS];

    // local thread indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0) - b2 * nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1) - b3 * nBBS1) + ly;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global T* oPtr = output +
                       (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) +
                       oInfo.strides[1] + 1;

// pull image to local memory
#pragma unroll
    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT;
         b += get_local_size(1), gy2 += get_local_size(1)) {
#pragma unroll
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH;
             a += get_local_size(0), gx2 += get_local_size(0)) {
            int x = gx2 - 1;
            int y = gy2 - 1;
            if (x >= 0 && x < oInfo.dims[0] && y >= 0 && y < oInfo.dims[1])
                outMem[b][a] =
                    oPtr[x * oInfo.strides[0] + y * oInfo.strides[1]];
            else
                outMem[b][a] = NOEDGE;
        }
    }

    int i = lx + 1;
    int j = ly + 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    int tid = get_local_id(0) + get_local_size(0) * get_local_id(1);

    int continueIter = 1;

    while (continueIter) {
        int cu = outMem[j][i];
        int nw = outMem[j - 1][i - 1];
        int no = outMem[j - 1][i];
        int ne = outMem[j - 1][i + 1];
        int ea = outMem[j][i + 1];
        int se = outMem[j + 1][i + 1];
        int so = outMem[j + 1][i];
        int sw = outMem[j + 1][i - 1];
        int we = outMem[j][i - 1];

        bool hasStrongNeighbour =
            nw == STRONG || no == STRONG || ne == STRONG || ea == STRONG ||
            se == STRONG || so == STRONG || sw == STRONG || we == STRONG;

        if (cu == WEAK && hasStrongNeighbour) outMem[j][i] = STRONG;

        barrier(CLK_LOCAL_MEM_FENCE);

        cu = outMem[j][i];

        bool _nw =
            outMem[j - 1][i - 1] == WEAK && VALID_BLOCK_IDX(j - 1, i - 1);
        bool _no = outMem[j - 1][i] == WEAK && VALID_BLOCK_IDX(j - 1, i);
        bool _ne =
            outMem[j - 1][i + 1] == WEAK && VALID_BLOCK_IDX(j - 1, i + 1);
        bool _ea = outMem[j][i + 1] == WEAK && VALID_BLOCK_IDX(j, i + 1);
        bool _se =
            outMem[j + 1][i + 1] == WEAK && VALID_BLOCK_IDX(j + 1, i + 1);
        bool _so = outMem[j + 1][i] == WEAK && VALID_BLOCK_IDX(j + 1, i);
        bool _sw =
            outMem[j + 1][i - 1] == WEAK && VALID_BLOCK_IDX(j + 1, i - 1);
        bool _we = outMem[j][i - 1] == WEAK && VALID_BLOCK_IDX(j, i - 1);

        bool hasWeakNeighbour =
            _nw || _no || _ne || _ea || _se || _so || _sw || _we;

        // Following Block is equivalent of __syncthreads_or in CUDA
        predicates[tid] = cu == STRONG && hasWeakNeighbour;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int nt = TOTAL_NUM_THREADS / 2; nt > 0; nt >>= 1) {
            if (tid < nt)
                predicates[tid] = predicates[tid] || predicates[tid + nt];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        continueIter = predicates[0];
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

    // Following Block is equivalent of __syncthreads_or in CUDA
    predicates[tid] = cu == STRONG && hasWeakNeighbour;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nt = TOTAL_NUM_THREADS / 2; nt > 0; nt >>= 1) {
        if (tid < nt) predicates[tid] = predicates[tid] || predicates[tid + nt];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    continueIter = predicates[0];

    if (continueIter > 0 && lx == 0 && ly == 0) atomic_add(hasChanged, 1);

    // Update output with shared memory result
    if (gx < (oInfo.dims[0] - 2) && gy < (oInfo.dims[1] - 2))
        oPtr[gx * oInfo.strides[0] + gy * oInfo.strides[1]] = outMem[j][i];
}
#endif

#if defined(SUPPRESS_LEFT_OVER)
__kernel void suppressLeftOverKernel(__global T* output, KParam oInfo,
                                     unsigned nBBS0, unsigned nBBS1) {
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx =
        get_local_size(0) * (get_group_id(0) - b2 * nBBS0) + get_local_id(0);
    const int gy =
        get_local_size(1) * (get_group_id(1) - b3 * nBBS1) + get_local_id(1);

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global T* oPtr = output +
                       (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) +
                       oInfo.strides[1] + 1;

    if (gx < (oInfo.dims[0] - 2) && gy < (oInfo.dims[1] - 2)) {
        int idx = gx * oInfo.strides[0] + gy * oInfo.strides[1];
        T val   = oPtr[idx];
        if (val == WEAK) oPtr[idx] = NOEDGE;
    }
}
#endif
