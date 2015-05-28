/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// OpenCL < 1.2 compatibility
#if !defined(__OPENCL_VERSION__) || __OPENCL_VERSION__ < 120
__inline unsigned popcount(unsigned x)
{
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x0000003F;
}
#endif

__kernel
void hamming_matcher_unroll(
    __global unsigned* out_idx,
    __global unsigned* out_dist,
    __global const T* query,
    KParam qInfo,
    __global const T* train,
    KParam tInfo,
    const unsigned max_dist,
    __local T* lmem)
{
    unsigned nquery = qInfo.dims[0];
    unsigned ntrain = tInfo.dims[0];

    unsigned f = get_global_id(0);
    unsigned tid = get_local_id(0);

    __local unsigned l_dist[THREADS];
    __local unsigned l_idx[THREADS];

    __local T* l_query = lmem;
    __local T* l_train = lmem + FEAT_LEN;

    l_dist[tid] = max_dist;
    l_idx[tid]  = 0xffffffff;

    bool valid_feat = (f < ntrain);

#ifdef USE_LOCAL_MEM
    if (valid_feat) {
        // Copy local_size(0) training features to shared memory
        #pragma unroll
        for (unsigned i = 0; i < FEAT_LEN; i++) {
            l_train[i * get_local_size(0) + tid] = train[i * ntrain + f];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int j = 0; j < (int)nquery; j++) {
        l_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < FEAT_LEN && valid_feat) {
            l_query[tid] = query[tid * nquery + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned dist = 0;
        if (valid_feat) {
            #pragma unroll
            for (int k = 0; k < (int)FEAT_LEN; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
#ifdef USE_LOCAL_MEM
                dist += popcount(l_train[k * get_local_size(0) + tid] ^ l_query[k]);
#else
                dist += popcount(train[k * ntrain + f] ^ l_query[k]);
#endif
            }
        }

        // Only stores the feature index and distance if it's smaller
        // than the best match found so far
        if (valid_feat) {
            l_dist[tid] = dist;
            l_idx[tid]  = f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Find best match in training features from block to the current
        // query feature
        if (tid < 128) {
            if (l_dist[tid + 128] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 128];
                l_idx[tid]  = l_idx[tid + 128];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 64) {
            if (l_dist[tid + 64] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 64];
                l_idx[tid]  = l_idx[tid + 64];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) {
            if (l_dist[tid + 32] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 32];
                l_idx[tid]  = l_idx[tid + 32];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) {
            if (l_dist[tid + 16] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 16];
                l_idx[tid]  = l_idx[tid + 16];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) {
            if (l_dist[tid + 8] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 8];
                l_idx[tid]  = l_idx[tid + 8];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) {
            if (l_dist[tid + 4] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 4];
                l_idx[tid]  = l_idx[tid + 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) {
            if (l_dist[tid + 2] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 2];
                l_idx[tid]  = l_idx[tid + 2];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) {
            if (l_dist[tid + 1] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 1];
                l_idx[tid]  = l_idx[tid + 1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * get_num_groups(0) + get_group_id(0)] = l_dist[0];
            out_idx[j * get_num_groups(0) + get_group_id(0)]  = l_idx[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void hamming_matcher(
    __global unsigned* out_idx,
    __global unsigned* out_dist,
    __global const T* query,
    KParam qInfo,
    __global const T* train,
    KParam tInfo,
    const unsigned max_dist,
    const unsigned feat_len,
    __local T* lmem)
{
    unsigned nquery = qInfo.dims[0];
    unsigned ntrain = tInfo.dims[0];

    unsigned f = get_global_id(0);
    unsigned tid = get_local_id(0);

    __local unsigned l_dist[THREADS];
    __local unsigned l_idx[THREADS];

    __local T* l_query = lmem;
    __local T* l_train = lmem + feat_len;

    l_dist[tid] = max_dist;
    l_idx[tid]  = 0xffffffff;

    bool valid_feat = (f < ntrain);

#ifdef USE_LOCAL_MEM
    if (valid_feat) {
        // Copy local_size(0) training features to shared memory
        for (unsigned i = 0; i < feat_len; i++) {
            l_train[i * get_local_size(0) + tid] = train[i * ntrain + f];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int j = 0; j < (int)nquery; j++) {
        l_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < feat_len && valid_feat) {
            l_query[tid] = query[tid * nquery + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned dist = 0;
        if (valid_feat) {
            for (int k = 0; k < (int)feat_len; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
#ifdef USE_LOCAL_MEM
                dist += popcount(l_train[k * get_local_size(0) + tid] ^ l_query[k]);
#else
                dist += popcount(train[k * ntrain + f] ^ l_query[k]);
#endif
            }
        }

        // Only stores the feature index and distance if it's smaller
        // than the best match found so far
        if (valid_feat) {
            l_dist[tid] = dist;
            l_idx[tid]  = f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Find best match in training features from block to the current
        // query feature
        if (tid < 128) {
            if (l_dist[tid + 128] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 128];
                l_idx[tid]  = l_idx[tid + 128];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 64) {
            if (l_dist[tid + 64] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 64];
                l_idx[tid]  = l_idx[tid + 64];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) {
            if (l_dist[tid + 32] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 32];
                l_idx[tid]  = l_idx[tid + 32];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) {
            if (l_dist[tid + 16] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 16];
                l_idx[tid]  = l_idx[tid + 16];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) {
            if (l_dist[tid + 8] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 8];
                l_idx[tid]  = l_idx[tid + 8];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) {
            if (l_dist[tid + 4] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 4];
                l_idx[tid]  = l_idx[tid + 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) {
            if (l_dist[tid + 2] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 2];
                l_idx[tid]  = l_idx[tid + 2];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) {
            if (l_dist[tid + 1] < l_dist[tid]) {
                l_dist[tid] = l_dist[tid + 1];
                l_idx[tid]  = l_idx[tid + 1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * get_num_groups(0) + get_group_id(0)] = l_dist[0];
            out_idx[j * get_num_groups(0) + get_group_id(0)]  = l_idx[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void select_matches(
    __global unsigned* idx,
    __global unsigned* dist,
    __global const unsigned* in_idx,
    __global const unsigned* in_dist,
    const unsigned nfeat,
    const unsigned nelem,
    const unsigned max_dist)
{
    unsigned f = get_global_id(0);
    unsigned lsz1 = get_local_size(1);
    unsigned sid = get_local_id(0) * lsz1 + get_local_id(1);

    __local unsigned l_dist[THREADS];
    __local unsigned l_idx[THREADS];

    bool valid_feat = (f < nfeat);

    if (valid_feat)
        l_dist[sid] = max_dist;
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned nelem_max = (nelem / lsz1) * lsz1;
    nelem_max = (nelem % lsz1 == 0) ? nelem_max : nelem_max + lsz1;

    for (unsigned i = get_local_id(1); i < nelem_max; i += get_local_size(1)) {
        if (valid_feat && i < nelem) {
            unsigned dist = in_dist[f * nelem + i];

            // Copy all best matches previously found in hamming_matcher() to
            // shared memory
            if (dist < l_dist[sid]) {
                l_dist[sid] = dist;
                l_idx[sid]  = in_idx[f * nelem + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned i = get_local_size(1) / 2; i > 0; i >>= 1) {
        if (get_local_id(1) < i) {
            if (valid_feat) {
                unsigned dist = l_dist[sid + i];
                if (dist < l_dist[sid]) {
                    l_dist[sid] = dist;
                    l_idx[sid]  = l_idx[sid + i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store best matches and indexes to training dataset
    if (get_local_id(1) == 0 && valid_feat) {
        dist[f] = l_dist[get_local_id(0) * get_local_size(1)];
        idx[f]  = l_idx[get_local_id(0) * get_local_size(1)];
    }
}
