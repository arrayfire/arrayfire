/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Reference pattern, generated for a patch size of 31x31, as suggested by
// original ORB paper
#define REF_PAT_SAMPLES 256
#define REF_PAT_COORDS 4
#define REF_PAT_LENGTH (REF_PAT_SAMPLES * REF_PAT_COORDS)

// Current reference pattern was borrowed from OpenCV, a randomly generated
// pattern will not achieve same quality as it must be trained like described
// in sections 4.2 and 4.3 of the original ORB paper.
__constant int ref_pat[] = {
    8,   -3,  9,   5,   4,   2,   7,   -12, -11, 9,   -8,  2,   7,   -12, 12,
    -13, 2,   -13, 2,   12,  1,   -7,  1,   6,   -2,  -10, -2,  -4,  -13, -13,
    -11, -8,  -13, -3,  -12, -9,  10,  4,   11,  9,   -13, -8,  -8,  -9,  -11,
    7,   -9,  12,  7,   7,   12,  6,   -4,  -5,  -3,  0,   -13, 2,   -12, -3,
    -9,  0,   -7,  5,   12,  -6,  12,  -1,  -3,  6,   -2,  12,  -6,  -13, -4,
    -8,  11,  -13, 12,  -8,  4,   7,   5,   1,   5,   -3,  10,  -3,  3,   -7,
    6,   12,  -8,  -7,  -6,  -2,  -2,  11,  -1,  -10, -13, 12,  -8,  10,  -7,
    3,   -5,  -3,  -4,  2,   -3,  7,   -10, -12, -6,  11,  5,   -12, 6,   -7,
    5,   -6,  7,   -1,  1,   0,   4,   -5,  9,   11,  11,  -13, 4,   7,   4,
    12,  2,   -1,  4,   4,   -4,  -12, -2,  7,   -8,  -5,  -7,  -10, 4,   11,
    9,   12,  0,   -8,  1,   -13, -13, -2,  -8,  2,   -3,  -2,  -2,  3,   -6,
    9,   -4,  -9,  8,   12,  10,  7,   0,   9,   1,   3,   7,   -5,  11,  -10,
    -13, -6,  -11, 0,   10,  7,   12,  1,   -6,  -3,  -6,  12,  10,  -9,  12,
    -4,  -13, 8,   -8,  -12, -13, 0,   -8,  -4,  3,   3,   7,   8,   5,   7,
    10,  -7,  -1,  7,   1,   -12, 3,   -10, 5,   6,   2,   -4,  3,   -10, -13,
    0,   -13, 5,   -13, -7,  -12, 12,  -13, 3,   -11, 8,   -7,  12,  -4,  7,
    6,   -10, 12,  8,   -9,  -1,  -7,  -6,  -2,  -5,  0,   12,  -12, 5,   -7,
    5,   3,   -10, 8,   -13, -7,  -7,  -4,  5,   -3,  -2,  -1,  -7,  2,   9,
    5,   -11, -11, -13, -5,  -13, -1,  6,   0,   -1,  5,   -3,  5,   2,   -4,
    -13, -4,  12,  -9,  -6,  -9,  6,   -12, -10, -8,  -4,  10,  2,   12,  -3,
    7,   12,  12,  12,  -7,  -13, -6,  5,   -4,  9,   -3,  4,   7,   -1,  12,
    2,   -7,  6,   -5,  1,   -13, 11,  -12, 5,   -3,  7,   -2,  -6,  7,   -8,
    12,  -7,  -13, -7,  -11, -12, 1,   -3,  12,  12,  2,   -6,  3,   0,   -4,
    3,   -2,  -13, -1,  -13, 1,   9,   7,   1,   8,   -6,  1,   -1,  3,   12,
    9,   1,   12,  6,   -1,  -9,  -1,  3,   -13, -13, -10, 5,   7,   7,   10,
    12,  12,  -5,  12,  9,   6,   3,   7,   11,  5,   -13, 6,   10,  2,   -12,
    2,   3,   3,   8,   4,   -6,  2,   6,   12,  -13, 9,   -12, 10,  3,   -8,
    4,   -7,  9,   -11, 12,  -4,  -6,  1,   12,  2,   -8,  6,   -9,  7,   -4,
    2,   3,   3,   -2,  6,   3,   11,  0,   3,   -3,  8,   -8,  7,   8,   9,
    3,   -11, -5,  -6,  -4,  -10, 11,  -5,  10,  -5,  -8,  -3,  12,  -10, 5,
    -9,  0,   8,   -1,  12,  -6,  4,   -6,  6,   -11, -10, 12,  -8,  7,   4,
    -2,  6,   7,   -2,  0,   -2,  12,  -5,  -8,  -5,  2,   7,   -6,  10,  12,
    -9,  -13, -8,  -8,  -5,  -13, -5,  -2,  8,   -8,  9,   -13, -9,  -11, -9,
    0,   1,   -8,  1,   -2,  7,   -4,  9,   1,   -2,  1,   -1,  -4,  11,  -6,
    12,  -11, -12, -9,  -6,  4,   3,   7,   7,   12,  5,   5,   10,  8,   0,
    -4,  2,   8,   -9,  12,  -5,  -13, 0,   7,   2,   12,  -1,  2,   1,   7,
    5,   11,  7,   -9,  3,   5,   6,   -8,  -13, -4,  -8,  9,   -5,  9,   -3,
    -3,  -4,  -7,  -3,  -12, 6,   5,   8,   0,   -7,  6,   -6,  12,  -13, 6,
    -5,  -2,  1,   -10, 3,   10,  4,   1,   8,   -4,  -2,  -2,  2,   -13, 2,
    -12, 12,  12,  -2,  -13, 0,   -6,  4,   1,   9,   3,   -6,  -10, -3,  -5,
    -3,  -13, -1,  1,   7,   5,   12,  -11, 4,   -2,  5,   -7,  -13, 9,   -9,
    -5,  7,   1,   8,   6,   7,   -8,  7,   6,   -7,  -4,  -7,  1,   -8,  11,
    -7,  -8,  -13, 6,   -12, -8,  2,   4,   3,   9,   10,  -5,  12,  3,   -6,
    -5,  -6,  7,   8,   -3,  9,   -8,  2,   -12, 2,   8,   -11, -2,  -10, 3,
    -12, -13, -7,  -9,  -11, 0,   -10, -5,  5,   -3,  11,  8,   -2,  -13, -1,
    12,  -1,  -8,  0,   9,   -13, -11, -12, -5,  -10, -2,  -10, 11,  -3,  9,
    -2,  -13, 2,   -3,  3,   2,   -9,  -13, -4,  0,   -4,  6,   -3,  -10, -4,
    12,  -2,  -7,  -6,  -11, -4,  9,   6,   -3,  6,   11,  -13, 11,  -5,  5,
    11,  11,  12,  6,   7,   -5,  12,  -2,  -1,  12,  0,   7,   -4,  -8,  -3,
    -2,  -7,  1,   -6,  7,   -13, -12, -8,  -13, -7,  -2,  -6,  -8,  -8,  5,
    -6,  -9,  -5,  -1,  -4,  5,   -13, 7,   -8,  10,  1,   5,   5,   -13, 1,
    0,   10,  -13, 9,   12,  10,  -1,  5,   -8,  10,  -9,  -1,  11,  1,   -13,
    -9,  -3,  -6,  2,   -1,  -10, 1,   12,  -13, 1,   -8,  -10, 8,   -11, 10,
    -6,  2,   -13, 3,   -6,  7,   -13, 12,  -9,  -10, -10, -5,  -7,  -10, -8,
    -8,  -13, 4,   -6,  8,   5,   3,   12,  8,   -13, -4,  2,   -3,  -3,  5,
    -13, 10,  -12, 4,   -13, 5,   -1,  -9,  9,   -4,  3,   0,   3,   3,   -9,
    -12, 1,   -6,  1,   3,   2,   4,   -8,  -10, -10, -10, 9,   8,   -13, 12,
    12,  -8,  -12, -6,  -5,  2,   2,   3,   7,   10,  6,   11,  -8,  6,   8,
    8,   -12, -7,  10,  -6,  5,   -3,  -9,  -3,  9,   -1,  -13, -1,  5,   -3,
    -7,  -3,  4,   -8,  -2,  -8,  3,   4,   2,   12,  12,  2,   -5,  3,   11,
    6,   -9,  11,  -13, 3,   -1,  7,   12,  11,  -1,  12,  4,   -3,  0,   -3,
    6,   4,   -11, 4,   12,  2,   -4,  2,   1,   -10, -6,  -8,  1,   -13, 7,
    -11, 1,   -13, 12,  -11, -13, 6,   0,   11,  -13, 0,   -1,  1,   4,   -13,
    3,   -9,  -2,  -9,  8,   -6,  -3,  -13, -6,  -8,  -2,  5,   -9,  8,   10,
    2,   7,   3,   -9,  -1,  -6,  -1,  -1,  9,   5,   11,  -2,  11,  -3,  12,
    -8,  3,   0,   3,   5,   -1,  4,   0,   10,  3,   -6,  4,   5,   -13, 0,
    -10, 5,   5,   8,   12,  11,  8,   9,   9,   -6,  7,   -4,  8,   -12, -10,
    4,   -10, 9,   7,   3,   12,  4,   9,   -7,  10,  -2,  7,   0,   12,  -2,
    -1,  -6,  0,   -11,
};

float block_reduce_sum(float val, local float* data) {
    unsigned idx = get_local_id(0) * get_local_size(0) + get_local_id(1);

    data[idx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned i = get_local_size(1) / 2; i > 0; i >>= 1) {
        if (get_local_id(1) < i) { data[idx] += data[idx + i]; }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return data[get_local_id(0) * get_local_size(0)];
}

kernel void keep_features(global float* x_out, __global float* y_out,
                            global float* score_out,
                            global const float* x_in,
                            global const float* y_in,
                            global const float* score_in,
                            global const unsigned* score_idx,
                            const unsigned n_feat) {
    unsigned f = get_global_id(0);

    if (f < n_feat) {
        x_out[f]     = x_in[score_idx[f]];
        y_out[f]     = y_in[score_idx[f]];
        score_out[f] = score_in[f];
    }
}

kernel void harris_response(
    global float* x_out, __global float* y_out, __global float* score_out,
    global const float* x_in, __global const float* y_in,
    const unsigned total_feat, global unsigned* usable_feat,
    global const T* image, KParam iInfo, const unsigned block_size,
    const float k_thr, const unsigned patch_size) {
    local float data[BLOCK_SIZE * BLOCK_SIZE];

    unsigned f = get_global_id(0);

    unsigned x, y;
    float ixx = 0.f, iyy = 0.f, ixy = 0.f;

    if (f < total_feat) {
        x = (unsigned)round(x_in[f]);
        y = (unsigned)round(y_in[f]);

        // Round feature size to nearest odd integer
        float size = 2.f * floor(patch_size / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned patch_r = ceil(size * sqrt(2.f) / 2.f);

        if (x >= patch_r && y >= patch_r && x < iInfo.dims[1] - patch_r &&
            y < iInfo.dims[0] - patch_r) {
            unsigned r = block_size / 2;

            unsigned block_size_sq = block_size * block_size;
            for (unsigned k = get_local_id(1); k < block_size_sq;
                 k += get_local_size(1)) {
                int i = k / block_size - r;
                int j = k % block_size - r;

                // Calculate local x and y derivatives
                float ix = image[(x + i + 1) * iInfo.dims[0] + y + j] -
                           image[(x + i - 1) * iInfo.dims[0] + y + j];
                float iy = image[(x + i) * iInfo.dims[0] + y + j + 1] -
                           image[(x + i) * iInfo.dims[0] + y + j - 1];

                // Accumulate second order derivatives
                ixx += ix * ix;
                iyy += iy * iy;
                ixy += ix * iy;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ixx = block_reduce_sum(ixx, data);
    iyy = block_reduce_sum(iyy, data);
    ixy = block_reduce_sum(ixy, data);

    if (f < total_feat && get_local_id(1) == 0) {
        unsigned idx = atomic_inc(usable_feat);
        if (idx < total_feat) {
            float tr  = ixx + iyy;
            float det = ixx * iyy - ixy * ixy;

            // Calculate Harris responses
            float resp = det - k_thr * (tr * tr);

            // Scale factor
            // TODO: improve scaling for responses
            float rscale = 0.001f;
            rscale       = rscale * rscale * rscale * rscale;

            x_out[idx]     = x;
            y_out[idx]     = y;
            score_out[idx] = resp * rscale;
        }
    }
}

kernel void centroid_angle(global const float* x_in,
                             global const float* y_in,
                             global float* orientation_out,
                             const unsigned total_feat, global const T* image,
                             KParam iInfo, const unsigned patch_size) {
    local float data[BLOCK_SIZE * BLOCK_SIZE];
    unsigned f = get_global_id(0);

    T m01 = (T)0, m10 = (T)0;

    if (f < total_feat) {
        unsigned x = (unsigned)round(x_in[f]);
        unsigned y = (unsigned)round(y_in[f]);

        unsigned r = patch_size / 2;

        if (x >= r && y >= r && x <= iInfo.dims[1] - r &&
            y <= iInfo.dims[0] - r) {
            unsigned patch_size_sq = patch_size * patch_size;
            for (unsigned k = get_local_id(1); k < patch_size_sq;
                 k += get_local_size(1)) {
                int i = k / patch_size - r;
                int j = k % patch_size - r;

                // Calculate first order moments
                T p = image[(x + i) * iInfo.dims[0] + y + j];
                m01 += j * p;
                m10 += i * p;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    m01 = block_reduce_sum(m01, data);
    m10 = block_reduce_sum(m10, data);

    if (f < total_feat && get_local_id(1) == 0) {
        float angle        = atan2(m01, m10);
        orientation_out[f] = angle;
    }
}

inline T get_pixel(unsigned x, unsigned y, const float ori, const unsigned size,
                   const int dist_x, const int dist_y, global const T* image,
                   KParam iInfo, const unsigned patch_size) {
    float ori_sin   = sin(ori);
    float ori_cos   = cos(ori);
    float patch_scl = (float)size / (float)patch_size;

    x += round(dist_x * patch_scl * ori_cos - dist_y * patch_scl * ori_sin);
    y += round(dist_x * patch_scl * ori_sin + dist_y * patch_scl * ori_cos);

    return image[x * iInfo.dims[0] + y];
}

kernel void extract_orb(global unsigned* desc_out, const unsigned n_feat,
                          global float* x_in, __global float* y_in,
                          global float* ori_in, __global float* size_out,
                          global const T* image, KParam iInfo,
                          const float scl, const unsigned patch_size) {
    unsigned f = get_global_id(0);

    unsigned x, y;

    if (f < n_feat) {
        x             = (unsigned)round(x_in[f]);
        y             = (unsigned)round(y_in[f]);
        float ori     = ori_in[f];
        unsigned size = patch_size;

        unsigned r = ceil(patch_size * sqrt(2.f) / 2.f);

        if (x >= r && y >= r && x < iInfo.dims[1] - r &&
            y < iInfo.dims[0] - r) {
            // Descriptor fixed at 256 bits for now
            for (unsigned i = get_local_id(1); i < 16; i += get_local_size(1)) {
                unsigned v = 0;

                for (unsigned j = 0; j < 16; j++) {
                    int dist_x = ref_pat[i * 16 * 4 + j * 4];
                    int dist_y = ref_pat[i * 16 * 4 + j * 4 + 1];
                    T p1 = get_pixel(x, y, ori, size, dist_x, dist_y, image,
                                     iInfo, patch_size);

                    dist_x = ref_pat[i * 16 * 4 + j * 4 + 2];
                    dist_y = ref_pat[i * 16 * 4 + j * 4 + 3];
                    T p2   = get_pixel(x, y, ori, size, dist_x, dist_y, image,
                                     iInfo, patch_size);

                    // Calculate bit based on p1 and p2 and shifts it to correct
                    // position
                    v |= (p1 < p2) << (j + 16 * (i % 2));
                }
                // Store 16 bits of descriptor
                atomic_add(&desc_out[f * 8 + i / 2], v);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (f < n_feat && get_local_id(1) == 0) {
        x_in[f]     = round(x * scl);
        y_in[f]     = round(y * scl);
        size_out[f] = patch_size * scl;
    }
}
