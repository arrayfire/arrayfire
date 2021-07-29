/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define STRONG 1
#define WEAK 2
#define NOEDGE 0

#if defined(INIT_EDGE_OUT)
kernel void initEdgeOutKernel(global T* output, KParam oInfo,
                              global const T* strong, KParam sInfo,
                              global const T* weak, KParam wInfo,
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
    global const T* wPtr =
        weak + (b2 * wInfo.strides[2] + b3 * wInfo.strides[3] + wInfo.offset) +
        wInfo.strides[1] + 1;

    global const T* sPtr =
        strong +
        (b2 * sInfo.strides[2] + b3 * sInfo.strides[3] + sInfo.offset) +
        sInfo.strides[1] + 1;

    global T* oPtr =
        output +
        (b2 * oInfo.strides[2] + b3 * oInfo.strides[3] + oInfo.offset) +
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
kernel void edgeTrackKernel(global T* output, KParam oInfo, unsigned nBBS0,
                            unsigned nBBS1, global volatile int* hasChanged) {
    // shared memory with 1 pixel border
    // strong and weak images are binary(char) images thus,
    // occupying only (16+2)*(16+2) = 324 bytes per shared memory tile
    local int outMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];
    local bool predicates[TOTAL_NUM_THREADS];

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
    global T* oPtr = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    // pull image to local memory
#pragma unroll
    for (int b = ly, gy2 = gy - 1; b < SHRD_MEM_HEIGHT;
         b += get_local_size(1), gy2 += get_local_size(1)) {
#pragma unroll
        for (int a = lx, gx2 = gx - 1; a < SHRD_MEM_WIDTH;
             a += get_local_size(0), gx2 += get_local_size(0)) {
            if (gx2 >= 0 && gx2 < oInfo.dims[0] && gy2 >= 0 &&
                gy2 < oInfo.dims[1])
                outMem[b][a] =
                    oPtr[gx2 * oInfo.strides[0] + gy2 * oInfo.strides[1]];
            else
                outMem[b][a] = NOEDGE;
        }
    }

    int i = lx + 1;
    int j = ly + 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    int tid = lx + get_local_size(0) * ly;

    bool continueIter = true;

    while (continueIter) {
        if (outMem[j][i] == WEAK) {
            int nw, no, ne, we, ea, sw, so, se;
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

        barrier(CLK_LOCAL_MEM_FENCE);

        predicates[tid] = false;
        if (outMem[j][i] == STRONG) {
            bool nw, no, ne, we, ea, sw, so, se;
            // clang-format off
            nw = outMem[j - 1][i - 1] == WEAK && VALID_BLOCK_IDX(j - 1, i - 1);
            no = outMem[j - 1][i]     == WEAK && VALID_BLOCK_IDX(j - 1, i);
            ne = outMem[j - 1][i + 1] == WEAK && VALID_BLOCK_IDX(j - 1, i + 1);
            we = outMem[j][i - 1]     == WEAK && VALID_BLOCK_IDX(j, i - 1);
            ea = outMem[j][i + 1]     == WEAK && VALID_BLOCK_IDX(j, i + 1);
            sw = outMem[j + 1][i - 1] == WEAK && VALID_BLOCK_IDX(j + 1, i - 1);
            so = outMem[j + 1][i]     == WEAK && VALID_BLOCK_IDX(j + 1, i);
            se = outMem[j + 1][i + 1] == WEAK && VALID_BLOCK_IDX(j + 1, i + 1);
            // clang-format on

            bool hasWeakNeighbour =
                nw || no || ne || ea || se || so || sw || we;

            predicates[tid] = hasWeakNeighbour;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Following Block is equivalent of __syncthreads_or in CUDA
        for (int nt = TOTAL_NUM_THREADS >> 1; nt > 0; nt >>= 1) {
            if (tid < nt) {
                predicates[tid] = predicates[tid] || predicates[tid + nt];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        continueIter = predicates[0];

        // Needed for Intel OpenCL implementation targeting CPUs
        barrier(CLK_LOCAL_MEM_FENCE);
    }

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

    if (continueIter && lx == 0 && ly == 0) atomic_inc(hasChanged);

    // Update output with shared memory result
    if (gx < (oInfo.dims[0] - 1) && gy < (oInfo.dims[1] - 1))
        oPtr[(gx * oInfo.strides[0] + gy * oInfo.strides[1])] = outMem[j][i];
}
#endif

#if defined(SUPPRESS_LEFT_OVER)
kernel void suppressLeftOverKernel(global T* output, KParam oInfo,
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
    global T* oPtr = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) +
                     oInfo.strides[1] + 1;

    if (gx < (oInfo.dims[0] - 2) && gy < (oInfo.dims[1] - 2)) {
        int idx = gx * oInfo.strides[0] + gy * oInfo.strides[1];
        T val   = oPtr[idx];
        if (val == WEAK) oPtr[idx] = NOEDGE;
    }
}
#endif
