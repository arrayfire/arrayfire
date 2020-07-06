/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// Output array is set to the following values during the progression
/// of the algorithm.
///
/// 0 - not processed
/// 1 - not valid
/// 2 - valid (candidate for neighborhood walk, pushed onto the queue)
///
/// Once, the algorithm is finished, output is reset
/// to either zero or \p newValue for all valid pixels.

#if defined(INIT_SEEDS)
kernel void init_seeds(global T *out, KParam oInfo, global const uint *seedsx,
                       KParam sxInfo, global const uint *seedsy,
                       KParam syInfo) {
    uint tid = get_global_id(0);
    if (tid < sxInfo.dims[0]) {
        uint x                          = seedsx[tid];
        uint y                          = seedsy[tid];
        out[(x + y * oInfo.strides[1])] = VALID;
    }
}
#endif

#if defined(FLOOD_FILL_STEP)

bool barrierOR(local int *predicates, const int workGroupSize) {
    int tid = get_local_id(0) + get_local_size(0) * get_local_id(1);
    for (int nt = workGroupSize >> 1; nt > 0; nt >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < nt) {
            predicates[tid] = (predicates[tid] + predicates[tid + nt]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return predicates[0] > 0;
}

kernel void flood_step(global T *out, KParam oInfo, global const T *img,
                       KParam iInfo, T lowValue, T highValue,
                       global volatile int *notFinished) {
    const int wgsize0 = get_local_size(0);
    const int wgsize1 = get_local_size(1);
#if AF_PLATFORM_NVIDIA
    const int FLOOD_LOOP_MAX_ITERATIONS =
        ceil(sqrt((float)(wgsize0 * wgsize0) + (float)(wgsize1 * wgsize1)));
#endif

    local T lmem[LMEM_HEIGHT][LMEM_WIDTH];
    local int predicates[GROUP_SIZE];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int d0 = oInfo.dims[0];
    const int d1 = oInfo.dims[1];
    const int s1 = oInfo.strides[1];

    for (int b = ly, gy2 = gy; b < LMEM_HEIGHT; b += wgsize1, gy2 += wgsize1) {
        for (int a = lx, gx2 = gx; a < LMEM_WIDTH;
             a += wgsize0, gx2 += wgsize0) {
            int x      = gx2 - RADIUS;
            int y      = gy2 - RADIUS;
            bool inROI = (x >= 0 && x < d0 && y >= 0 && y < d1);
            lmem[b][a] = (inROI ? out[x + y * s1] : INVALID);
        }
    }
    int i = lx + RADIUS;
    int j = ly + RADIUS;

    T tImgVal =
        img[(clamp(gx, 0, (int)(iInfo.dims[0] - 1)) * iInfo.strides[0] +
             clamp(gy, 0, (int)(iInfo.dims[1] - 1)) * iInfo.strides[1]) +
            iInfo.offset];
    const int isPxBtwnThresholds =
        (tImgVal >= lowValue && tImgVal <= highValue);

    int tid = lx + wgsize0 * ly;

    predicates[tid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    T origOutVal      = lmem[j][i];
    T centerVal       = origOutVal;
    bool blockChanged = false;
    bool isBorderPxl =
        (lx == 0 || ly == 0 || lx == (wgsize0 - 1) || ly == (wgsize1 - 1));
    int loopCount = 0;
    do {
        int validNeighbors = 0;
        for (int no_j = -RADIUS; no_j <= RADIUS; ++no_j) {
            for (int no_i = -RADIUS; no_i <= RADIUS; ++no_i) {
                T currVal = lmem[j + no_j][i + no_i];
                validNeighbors += (currVal == VALID);
            }
        }
        // Exempt current/center pixel from validNeighbors
        validNeighbors -= (centerVal == VALID);
        barrier(CLK_LOCAL_MEM_FENCE);

        bool outChanged = (lmem[j][i] == ZERO && (validNeighbors > 0));
        if (outChanged) { lmem[j][i] = (T)(isPxBtwnThresholds + INVALID); }
        predicates[tid] = (int)(outChanged);
        blockChanged    = barrierOR(predicates, GROUP_SIZE);
        centerVal       = lmem[j][i];
#if AF_PLATFORM_NVIDIA
        // NVIDIA OpenCL seems to have a bug with while loop, where it
        // is somehow going into infinite loop even if the loop should
        // This isn't an issue on AMD as expected.
    } while (blockChanged && (loopCount++ < FLOOD_LOOP_MAX_ITERATIONS));
#else
    } while (blockChanged);
#endif

    bool brdrChngd =
        (isBorderPxl && centerVal != origOutVal && centerVal == VALID);
    predicates[tid] = (int)(brdrChngd);

    brdrChngd = barrierOR(predicates, GROUP_SIZE);

    if (brdrChngd && lx == 0 && ly == 0) {
        // Atleast one border pixel changed. Therefore, mark for
        // another kernel launch to propogate changes beyond border
        // of this block
        atomic_inc(notFinished);
    }

    if (gx < d0 && gy < d1) { out[(gx + gy * s1)] = lmem[j][i]; }
}
#endif

#if defined(FINALIZE_OUTPUT)
kernel void finalize_output(global T *out, KParam oInfo, T newValue) {
    uint gx = get_global_id(0);
    uint gy = get_global_id(1);
    if (gx < oInfo.dims[0] && gy < oInfo.dims[1]) {
        uint idx = gx + gy * oInfo.strides[1];
        T val    = out[idx];
        out[idx] = (val == VALID ? newValue : ZERO);
    }
}
#endif
