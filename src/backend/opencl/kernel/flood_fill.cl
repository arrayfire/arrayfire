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
        uint x                                             = seedsx[tid];
        uint y                                             = seedsy[tid];
        out[(x * oInfo.strides[0] + y * oInfo.strides[1])] = VALID;
    }
}
#endif

#if defined(FLOOD_FILL_STEP)

int barrierOR(local int *predicates) {
    int tid = get_local_id(0) + get_local_size(0) * get_local_id(1);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nt = GROUP_SIZE / 2; nt > 0; nt >>= 1) {
        if (tid < nt) {
            predicates[tid] = (predicates[tid] | predicates[tid + nt]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int retVal = predicates[0];
#if AF_IS_PLATFORM_NVIDIA
    // Without the extra barrier sync after reading the reduction result,
    // the caller's loop is going into infinite loop occasionally which is
    // in turn randoms hangs. This doesn't seem to be an issue on non-nvidia
    // hardware. Hence, the check.
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    return retVal;
}

kernel void flood_step(global T *out, KParam oInfo, global const T *img,
                       KParam iInfo, T lowValue, T highValue,
                       global volatile int *notFinished) {
    local T lmem[LMEM_HEIGHT][LMEM_WIDTH];
    local int predicates[GROUP_SIZE];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int d0 = oInfo.dims[0];
    const int d1 = oInfo.dims[1];
    const int s0 = oInfo.strides[0];
    const int s1 = oInfo.strides[1];

    for (int b = ly, gy2 = gy; b < LMEM_HEIGHT;
         b += get_local_size(1), gy2 += get_local_size(1)) {
        for (int a = lx, gx2 = gx; a < LMEM_WIDTH;
             a += get_local_size(0), gx2 += get_local_size(0)) {
            int x      = gx2 - RADIUS;
            int y      = gy2 - RADIUS;
            bool inROI = (x >= 0 && x < d0 && y >= 0 && y < d1);
            lmem[b][a] = (inROI ? out[x * s0 + y * s1] : INVALID);
        }
    }
    int i = lx + RADIUS;
    int j = ly + RADIUS;

    T tImgVal =
        img[(clamp(gx, 0, (int)(iInfo.dims[0] - 1)) * iInfo.strides[0] +
             clamp(gy, 0, (int)(iInfo.dims[1] - 1)) * iInfo.strides[1])];
    const int isPxBtwnThresholds =
        (tImgVal >= lowValue && tImgVal <= highValue);

    int tid = lx + get_local_size(0) * ly;

    barrier(CLK_LOCAL_MEM_FENCE);

    T origOutVal     = lmem[j][i];
    bool isBorderPxl = (lx == 0 || ly == 0 || lx == (get_local_size(0) - 1) ||
                        ly == (get_local_size(1) - 1));

    for (bool blkChngd = true; blkChngd; blkChngd = barrierOR(predicates)) {
        int validNeighbors = 0;
        for (int no_j = -RADIUS; no_j <= RADIUS; ++no_j) {
            for (int no_i = -RADIUS; no_i <= RADIUS; ++no_i) {
                T currVal = lmem[j + no_j][i + no_i];
                validNeighbors += (currVal == VALID);
            }
        }
        bool outChanged = (lmem[j][i] == ZERO && (validNeighbors > 0));
        predicates[tid] = outChanged;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (outChanged) { lmem[j][i] = (T)(isPxBtwnThresholds + INVALID); }
    }

    T newOutVal = lmem[j][i];

    bool brdrChngd =
        (isBorderPxl && newOutVal != origOutVal && newOutVal == VALID);
    predicates[tid] = brdrChngd;

    brdrChngd = barrierOR(predicates) > 0;

    if (gx < d0 && gy < d1) {
        if (brdrChngd && lx == 0 && ly == 0) {
            // Atleast one border pixel changed. Therefore, mark for
            // another kernel launch to propogate changes beyond border
            // of this block
            atomic_inc(notFinished);
        }
        out[(gx * s0 + gy * s1)] = lmem[j][i];
    }
}
#endif

#if defined(FINALIZE_OUTPUT)
kernel void finalize_output(global T *out, KParam oInfo, T newValue) {
    uint gx = get_global_id(0);
    uint gy = get_global_id(1);
    if (gx < oInfo.dims[0] && gy < oInfo.dims[1]) {
        uint idx = gx * oInfo.strides[0] + gy * oInfo.strides[1];
        T val    = out[idx];
        out[idx] = (val == VALID ? newValue : ZERO);
    }
}
#endif
