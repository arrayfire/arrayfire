/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Exchange trick: Morgan McGuire, ShaderX 2008
#define swap(a, b)           \
    {                        \
        T tmp = a;           \
        a     = min(a, b);   \
        b     = max(tmp, b); \
    }

void load2ShrdMem_1d(local T* shrd, global const T* in, int lx, int dim0,
                     int gx, int inStride0) {
    if (pad == AF_PAD_ZERO) {
        if (gx < 0 || gx >= dim0)
            shrd[lx] = (T)0;
        else
            shrd[lx] = in[gx];
    } else if (pad == AF_PAD_SYM) {
        if (gx < 0) gx *= -1;
        if (gx >= dim0) gx = 2 * (dim0 - 1) - gx;
        shrd[lx] = in[gx];
    }
}

kernel void medfilt1(global T* out, KParam oInfo, __global const T* in,
                       KParam iInfo, local T* localMem, int nBBS0) {
    // calculate necessary offset and window parameters
    const int padding = w_wid - 1;
    const int halo    = padding / 2;
    const int shrdLen = get_local_size(0) + padding;

    // batch offsets
    unsigned b1            = get_group_id(0) / nBBS0;
    unsigned b0            = get_group_id(0) - b1 * nBBS0;
    unsigned b2            = get_group_id(1);
    unsigned b3            = get_group_id(2);
    global const T* iptr = in +
                             (b1 * iInfo.strides[1] + b2 * iInfo.strides[2] +
                              b3 * iInfo.strides[3]) +
                             iInfo.offset;
    global T* optr = out +
                       (b1 * oInfo.strides[1] + b2 * oInfo.strides[2] +
                        b3 * oInfo.strides[3]) +
                       oInfo.offset;

    // local neighborhood indices
    int lx = get_local_id(0);

    // global indices
    int gx = get_local_size(0) * b0 + lx;

    int s0 = iInfo.strides[0];
    int d0 = iInfo.dims[0];
    for (int a = lx, gx2 = gx; a < shrdLen;
         a += get_local_size(0), gx2 += get_local_size(0)) {
        load2ShrdMem_1d(localMem, iptr, a, d0, gx2 - halo, s0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Only continue if we're at a valid location
    if (gx < iInfo.dims[0]) {
        // pull top half from shared memory into local memory
        T v[ARR_SIZE];

#pragma unroll
        for (int k = 0; k <= w_wid / 2 + 1; k++) { v[k] = localMem[lx + k]; }

        // with each pass, remove min and max values and add new value
        // initial sort
        // ensure min in first half, max in second half
#pragma unroll
        for (int i = 0; i < ARR_SIZE / 2; i++) {
            swap(v[i], v[ARR_SIZE - 1 - i]);
        }
        // move min in first half to first pos
#pragma unroll
        for (int i = 1; i < (ARR_SIZE + 1) / 2; i++) { swap(v[0], v[i]); }
        // move max in second half to last pos
#pragma unroll
        for (int i = ARR_SIZE - 2; i >= ARR_SIZE / 2; i--) {
            swap(v[i], v[ARR_SIZE - 1]);
        }

        int last = ARR_SIZE - 1;

        for (int k = w_wid / 2 + 2; k < w_wid; k++) {
            // add new contestant to first position in array
            v[0] = localMem[lx + k];

            last--;

            // place max in last half, min in first half
            for (int i = 0; i < (last + 1) / 2; i++) {
                swap(v[i], v[last - i]);
            }
            // now perform swaps on each half such that
            // max is in last pos, min is in first pos
            for (int i = 1; i <= last / 2; i++) { swap(v[0], v[i]); }
            for (int i = last - 1; i >= (last + 1) / 2; i--) {
                swap(v[i], v[last]);
            }
        }

        // no more new contestants
        // may still have to sort the last row
        // each outer loop drops the min and max
        for (int k = 0; k < last; k++) {
            // move max/min into respective halves
            for (int i = k; i < ARR_SIZE / 2; i++) {
                swap(v[i], v[ARR_SIZE - 1 - i]);
            }
            // move min into first pos
            for (int i = k + 1; i <= ARR_SIZE / 2; i++) { swap(v[k], v[i]); }
            // move max into last pos
            for (int i = ARR_SIZE - k - 2; i >= ARR_SIZE / 2; i--) {
                swap(v[i], v[ARR_SIZE - 1 - k]);
            }
        }

        // pick the middle element of the first row
        optr[gx * oInfo.strides[0]] = v[last / 2];
    }
}
