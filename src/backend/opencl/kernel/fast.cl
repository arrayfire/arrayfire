/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define MAX_VAL(A,B) (A<B) ? (B) : (A)

inline int idx_y(const int i)
{
    int j = i - 4;
    int k = min(j, 8 - j);
    return clamp(k, -3, 3);
}

inline int idx_x(const int i)
{
    return idx_y((i + 4) & 15);
}

inline int idx(const int x, const int y)
{
    return ((get_local_id(0) + 3 + x) + (get_local_size(0) + 6) * (get_local_id(1) + 3 + y));
}

// test_greater()
// Tests if a pixel x > p + thr
inline int test_greater(const float x, const float p, const float thr)
{
    return (x >= p + thr);
}

// test_smaller()
// Tests if a pixel x < p - thr
inline int test_smaller(const float x, const float p, const float thr)
{
    return (x <= p - thr);
}

// test_pixel()
// Returns -1 when x < p - thr
// Returns  0 when x >= p - thr && x <= p + thr
// Returns  1 when x > p + thr
inline int test_pixel(__local T* local_image, const float p, const float thr, const int x, const int y)
{
    return -test_smaller((float)local_image[idx(x,y)], p, thr) | test_greater((float)local_image[idx(x,y)], p, thr);
}

void locate_features_core(
    __local T* local_image,
    __global float* score,
    KParam iInfo,
    const float thr,
    int x, int y,
    const unsigned edge)
{
    if (x >= iInfo.dims[0] - edge || y >= iInfo.dims[1] - edge) return;

    float p = local_image[idx( 0, 0)];

    // Start by testing opposite pixels of the circle that will result in
    // a non-kepoint
    int d = test_pixel(local_image, p, thr, -3,  0) | test_pixel(local_image, p, thr, 3,  0);
    if (d == 0)
        return;

    d &= test_pixel(local_image, p, thr, -2,  2) | test_pixel(local_image, p, thr,  2, -2);
    d &= test_pixel(local_image, p, thr,  0,  3) | test_pixel(local_image, p, thr,  0, -3);
    d &= test_pixel(local_image, p, thr,  2,  2) | test_pixel(local_image, p, thr, -2, -2);
    if (d == 0)
        return;

    d &= test_pixel(local_image, p, thr, -3,  1) | test_pixel(local_image, p, thr,  3, -1);
    d &= test_pixel(local_image, p, thr, -1,  3) | test_pixel(local_image, p, thr,  1, -3);
    d &= test_pixel(local_image, p, thr,  1,  3) | test_pixel(local_image, p, thr, -1, -3);
    d &= test_pixel(local_image, p, thr,  3,  1) | test_pixel(local_image, p, thr, -3, -1);
    if (d == 0)
        return;

    int sum = 0;

    // Sum responses [-1, 0 or 1] of first ARC_LENGTH pixels
    for (int i = 0; i < ARC_LENGTH; i++)
        sum += test_pixel(local_image, p, thr, idx_x(i), idx_y(i));

    // Test maximum and mininmum responses of first segment of ARC_LENGTH
    // pixels
    int max_sum = 0, min_sum = 0;
    max_sum = max(max_sum, sum);
    min_sum = min(min_sum, sum);

    // Sum responses and test the remaining 16-ARC_LENGTH pixels of the circle
    for (int i = ARC_LENGTH; i < 16; i++) {
        sum -= test_pixel(local_image, p, thr, idx_x(i-ARC_LENGTH), idx_y(i-ARC_LENGTH));
        sum += test_pixel(local_image, p, thr, idx_x(i), idx_y(i));
        max_sum = max(max_sum, sum);
        min_sum = min(min_sum, sum);
    }

    // To completely test all possible segments, it's necessary to test
    // segments that include the top junction of the circle
    for (int i = 0; i < ARC_LENGTH-1; i++) {
        sum -= test_pixel(local_image, p, thr, idx_x(16-ARC_LENGTH+i), idx_y(16-ARC_LENGTH+i));
        sum += test_pixel(local_image, p, thr, idx_x(i), idx_y(i));
        max_sum = max(max_sum, sum);
        min_sum = min(min_sum, sum);
    }

    // If sum at some point was equal to (+-)ARC_LENGTH, there is a segment
    // for which all pixels are much brighter or much darker than central
    // pixel p.
    if (max_sum == ARC_LENGTH || min_sum == -ARC_LENGTH) {
        // Compute scores for brighter and darker pixels
        float s_bright = 0, s_dark = 0;
        for (int i = 0; i < 16; i++) {
            float p_x    = local_image[idx(idx_x(i), idx_y(i))];
            float weight = fabs((float)p_x - (float)p) - thr;
            s_bright += test_greater(p_x, p, thr) * weight;
            s_dark   += test_smaller(p_x, p, thr) * weight;
        }

        score[x + iInfo.dims[0] * y] = MAX_VAL(s_bright, s_dark);
    }
}

void load_shared_image(
    __global const T *in,
    KParam iInfo,
    __local T *local_image,
    unsigned ix, unsigned iy,
    unsigned bx, unsigned by,
    unsigned  x, unsigned  y,
    unsigned lx, unsigned ly)
{
    // Copy an image patch to shared memory, with a 3-pixel edge
    if (ix < lx && iy < ly && x - 3 < iInfo.dims[0] && y - 3 < iInfo.dims[1]) {
        local_image[(ix)      + (bx+6) * (iy)]    = in[(x-3)    + iInfo.dims[0] * (y-3)];
        if (x + lx - 3 < iInfo.dims[0])
            local_image[(ix + lx) + (bx+6) * (iy)]    = in[(x+lx-3) + iInfo.dims[0] * (y-3)];
        if (y + ly - 3 < iInfo.dims[1])
            local_image[(ix)      + (bx+6) * (iy+ly)] = in[(x-3)    + iInfo.dims[0] * (y+ly-3)];
        if (x + lx - 3 < iInfo.dims[0] && y + ly - 3 < iInfo.dims[1])
            local_image[(ix + lx) + (bx+6) * (iy+ly)] = in[(x+lx-3) + iInfo.dims[0] * (y+ly-3)];
    }
}

__kernel
void locate_features(
    __global const T* in,
    KParam          iInfo,
    __global float* score,
    const float thr,
    const unsigned edge,
    __local T* local_image)
{
    unsigned ix = get_local_id(0);
    unsigned iy = get_local_id(1);
    unsigned bx = get_local_size(0);
    unsigned by = get_local_size(1);
    unsigned x = bx * get_group_id(0) + ix + edge;
    unsigned y = by * get_group_id(1) + iy + edge;
    unsigned lx = bx / 2 + 3;
    unsigned ly = by / 2 + 3;

    load_shared_image(in, iInfo, local_image, ix, iy, bx, by, x, y, lx, ly);
    barrier(CLK_LOCAL_MEM_FENCE);
    locate_features_core(local_image, score,
                         iInfo, thr, x, y, edge);
}

__kernel
void non_max_counts(
    __global unsigned *d_counts,
    __global unsigned *d_offsets,
    __global unsigned *d_total,
    __global float *flags,
    __global const float* score,
    KParam iInfo,
    const unsigned edge)
{
    __local unsigned s_counts[256];

    const int yid = get_group_id(1) * get_local_size(1) * 8 + get_local_id(1);
    const int yend = (get_group_id(1) + 1) * get_local_size(1) * 8;
    const int yoff = get_local_size(1);

    unsigned count = 0;

    const int max1 = (int)iInfo.dims[1] - edge - 1;
    for (int y = yid; y < yend; y += yoff) {
        if (y >= max1 || y <= (int)(edge+1)) continue;

        const int xid = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);
        const int xend = (get_group_id(0) + 1) * get_local_size(0) * 2;

        const int max0 = (int)iInfo.dims[0] - edge - 1;
        for (int x = xid; x < xend; x += get_local_size(0)) {
            if (x >= max0 || x <= (int)(edge+1)) continue;

            float v = score[y * iInfo.dims[0] + x];
            if (v == 0) {
#if NONMAX
                flags[y * iInfo.dims[0] + x] = 0;
#endif
                continue;
            }

#if NONMAX
                float max_v = v;
                max_v = MAX_VAL(score[x-1 + iInfo.dims[0] * (y-1)], score[x-1 + iInfo.dims[0] * y]);
                max_v = MAX_VAL(max_v, score[x-1 + iInfo.dims[0] * (y+1)]);
                max_v = MAX_VAL(max_v, score[x   + iInfo.dims[0] * (y-1)]);
                max_v = MAX_VAL(max_v, score[x   + iInfo.dims[0] * (y+1)]);
                max_v = MAX_VAL(max_v, score[x+1 + iInfo.dims[0] * (y-1)]);
                max_v = MAX_VAL(max_v, score[x+1 + iInfo.dims[0] * (y)  ]);
                max_v = MAX_VAL(max_v, score[x+1 + iInfo.dims[0] * (y+1)]);

                v = (v > max_v) ? v : 0;
                flags[y * iInfo.dims[0] + x] = v;
                if (v == 0) continue;
#endif

            count++;
        }
    }

    const int tid = get_local_size(0) * get_local_id(1) + get_local_id(0);

    s_counts[tid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128) s_counts[tid] += s_counts[tid + 128]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  64) s_counts[tid] += s_counts[tid +  64]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  32) s_counts[tid] += s_counts[tid +  32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  16) s_counts[tid] += s_counts[tid +  16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   8) s_counts[tid] += s_counts[tid +   8]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   4) s_counts[tid] += s_counts[tid +   4]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   2) s_counts[tid] += s_counts[tid +   2]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   1) s_counts[tid] += s_counts[tid +   1]; barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        const int bid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        unsigned total = s_counts[0] ? atomic_add(d_total, s_counts[0]) : 0;
        d_counts [bid] = s_counts[0];
        d_offsets[bid] = total;
    }
}

__kernel void get_features(
    __global float* x_out,
    __global float* y_out,
    __global float* score_out,
    __global const float* flags,
    __global const unsigned* d_counts,
    __global const unsigned* d_offsets,
    KParam iInfo,
    const unsigned total,
    const unsigned edge)
{
    const int xid = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);
    const int yid = get_group_id(1) * get_local_size(1) * 8 + get_local_id(1);
    const int tid = get_local_size(0) * get_local_id(1) + get_local_id(0);

    const int xoff = get_local_size(0);
    const int yoff = get_local_size(1);

    const int xend = (get_group_id(0) + 1) * get_local_size(0) * 2;
    const int yend = (get_group_id(1) + 1) * get_local_size(1) * 8;

    const int bid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    __local unsigned s_count;
    __local unsigned s_idx;

    if (tid == 0) {
        s_count  = d_counts [bid];
        s_idx    = d_offsets[bid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Blocks that are empty, please bail
    if (s_count == 0) return;
    for (int y = yid; y < yend; y += yoff) {
        if (y >= iInfo.dims[1] - edge - 1 || y <= edge+1) continue;
        for (int x = xid; x < xend; x += xoff) {
            if (x >= iInfo.dims[0] - edge - 1 || x <= edge+1) continue;

            float v = flags[y * iInfo.dims[0] + x];
            if (v == 0) continue;

            unsigned id = atomic_inc(&s_idx);
            if (id < total) {
                y_out[id] = x;
                x_out[id] = y;
                score_out[id] = v;
            }
        }
    }
}
