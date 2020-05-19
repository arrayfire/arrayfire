/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if AF_BORDER_TYPE == AF_PAD_SYM

int trimIndex(int idx, const int len) {
    int ret_val = idx;
    int offset  = abs(ret_val) % len;
    if (ret_val < 0) {
        int offset = (abs(ret_val) - 1) % len;
        ret_val    = offset;
    } else if (ret_val >= len) {
        int offset = abs(ret_val) % len;
        ret_val    = len - offset - 1;
    }
    return ret_val;
}

// TODO(Pradeep) move trimindex from all locations into
//              a single header after opencl cache is cleaned up
int idxByndEdge(const int i, const int lb, const int len) {
    return trimIndex(i - lb, len);
}

#elif AF_BORDER_TYPE == AF_PAD_CLAMP_TO_EDGE

int idxByndEdge(const int i, const int lb, const int len) {
    return clamp(i - lb, 0, len - 1);
}

#elif AF_BORDER_TYPE == AF_PAD_PERIODIC

int idxByndEdge(const int i, const int lb, const int len) {
    int rem  = (i - lb) % len;
    int cond = rem < 0;
    return cond * (rem + len) + (1 - cond) * rem;
}

#else

#define DEFAULT_BORDER

#endif

kernel void padBorders(global T* out, KParam oInfo, __global const T* in,
                         KParam iInfo, int l0, int l1, int l2, int l3,
                         unsigned blk_x, unsigned blk_y) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int k  = get_group_id(0) / blk_x;
    const int l  = get_group_id(1) / blk_y;

    const int blockIdx_x = get_group_id(0) - (blk_x)*k;
    const int blockIdx_y = get_group_id(1) - (blk_y)*l;
    const int i          = blockIdx_x * get_local_size(0) + lx;
    const int j          = blockIdx_y * get_local_size(1) + ly;

    const int d0 = iInfo.dims[0];
    const int d1 = iInfo.dims[1];
    const int d2 = iInfo.dims[2];
    const int d3 = iInfo.dims[3];
    const int s0 = iInfo.strides[0];
    const int s1 = iInfo.strides[1];
    const int s2 = iInfo.strides[2];
    const int s3 = iInfo.strides[3];

    global const T* src = in + iInfo.offset;
    global T* dst       = out;

    bool isNotPadding =
        (l >= l3 && l < (d3 + l3)) && (k >= l2 && k < (d2 + l2)) &&
        (j >= l1 && j < (d1 + l1)) && (i >= l0 && i < (d0 + l0));
    T value = (T)0;

    if (isNotPadding) {
        unsigned iLOff = (l - l3) * s3;
        unsigned iKOff = (k - l2) * s2;
        unsigned iJOff = (j - l1) * s1;
        unsigned iIOff = (i - l0) * s0;

        value = src[iLOff + iKOff + iJOff + iIOff];
    } else {
#if !defined(DEFAULT_BORDER)
        unsigned iLOff = idxByndEdge(l, l3, d3) * s3;
        unsigned iKOff = idxByndEdge(k, l2, d2) * s2;
        unsigned iJOff = idxByndEdge(j, l1, d1) * s1;
        unsigned iIOff = idxByndEdge(i, l0, d0) * s0;

        value = src[iLOff + iKOff + iJOff + iIOff];
#endif
    }

    if (i < oInfo.dims[0] && j < oInfo.dims[1] && k < oInfo.dims[2] &&
        l < oInfo.dims[3]) {
        unsigned offset = l * oInfo.strides[3] + k * oInfo.strides[2] +
                          j * oInfo.strides[1] + i * oInfo.strides[0];
        dst[offset] = value;
    }
}
