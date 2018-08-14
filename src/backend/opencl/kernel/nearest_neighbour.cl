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

#ifdef USE_DOUBLE
To _sad_(T v1, T v2)
{
    return fabs(v1 - v2);
}
#else
To _sad_(T v1, T v2)
{
    return fabs((float)v1 - (float)v2);
}
#endif

To _ssd_(T v1, T v2)
{
    return (v1 - v2) * (v1 - v2);
}

#ifdef __SHD__
unsigned _shd_(T v1, T v2)
{
    return popcount(v1 ^ v2);
}
#endif

__kernel
void nearest_neighbour_unroll(
    __global To* out_dist,
    __global const T* query,
    KParam qInfo,
    __global const T* train,
    KParam tInfo,
    const To max_dist,
    __local T* lmem)
{
    unsigned nquery = qInfo.dims[0];
    unsigned ntrain = tInfo.dims[0];

    unsigned f = get_global_id(0);
    unsigned tid = get_local_id(0);

    __local To l_dist[THREADS];

    __local T* l_query = lmem;
    __local T* l_train = lmem + FEAT_LEN;

    l_dist[tid] = max_dist;

    bool valid_feat = (f < ntrain);

#ifdef USE_LOCAL_MEM
    if (valid_feat) {
        // Copy local_size(0) training features to shared memory
        #pragma unroll
        for (unsigned i = 0; i < FEAT_LEN; i++) {
            l_train[i * get_local_size(0) + tid] = train[i * ntrain + f + tInfo.offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int j = 0; j < (int)nquery; j++) {
        l_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < FEAT_LEN) {
            l_query[tid] = query[tid * nquery + j + qInfo.offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        To dist = 0;
        if (valid_feat) {
            #pragma unroll
            for (int k = 0; k < (int)FEAT_LEN; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
#ifdef USE_LOCAL_MEM
                dist += DISTOP(l_train[k * get_local_size(0) + tid], l_query[k]);
#else
                dist += DISTOP(train[k * ntrain + f + tInfo.offset], l_query[k]);
#endif
            }
        }

        // Only stores the feature index and distance if it's smaller
        // than the best match found so far
        if (valid_feat) {
            l_dist[tid] = dist;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * ntrain + f] = l_dist[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void nearest_neighbour(
    __global To* out_dist,
    __global const T* query,
    KParam qInfo,
    __global const T* train,
    KParam tInfo,
    const To max_dist,
    const unsigned feat_len,
    __local T* lmem)
{
    unsigned nquery = qInfo.dims[0];
    unsigned ntrain = tInfo.dims[0];

    unsigned f = get_global_id(0);
    unsigned tid = get_local_id(0);

    __local To l_dist[THREADS];

    __local T* l_query = lmem;
    __local T* l_train = lmem + feat_len;

    l_dist[tid] = max_dist;

    bool valid_feat = (f < ntrain);

#ifdef USE_LOCAL_MEM
    if (valid_feat) {
        // Copy local_size(0) training features to shared memory
        for (unsigned i = 0; i < feat_len; i++) {
            l_train[i * get_local_size(0) + tid] = train[i * ntrain + f + tInfo.offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int j = 0; j < (int)nquery; j++) {
        l_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < feat_len) {
            l_query[tid] = query[tid * nquery + j + qInfo.offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        To dist = 0;
        if (valid_feat) {
            for (int k = 0; k < (int)feat_len; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
#ifdef USE_LOCAL_MEM
                dist += DISTOP(l_train[k * get_local_size(0) + tid], l_query[k]);
#else
                dist += DISTOP(train[k * ntrain + f + tInfo.offset], l_query[k]);
#endif
            }
        }

        // Only stores the feature index and distance if it's smaller
        // than the best match found so far
        if (valid_feat) {
            l_dist[tid] = dist;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * ntrain + f] = l_dist[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void select_matches(
    __global unsigned* idx,
    __global To* dist,
    __global const unsigned* in_idx,
    __global const To* in_dist,
    const unsigned nfeat,
    const unsigned nelem,
    const To max_dist)
{
    unsigned f = get_global_id(0);
    unsigned lsz1 = get_local_size(1);
    unsigned sid = get_local_id(0) * lsz1 + get_local_id(1);

    __local To l_dist[THREADS];
    __local unsigned l_idx[THREADS];

    bool valid_feat = (f < nfeat);

    l_dist[sid] = max_dist;
    if (valid_feat) {
        for (unsigned i = get_local_id(1); i < nelem; i += get_local_size(1)) {
            To dist = in_dist[f * nelem + i];

            // Copy all best matches previously found in nearest_neighbour() to
            // shared memory
            if (dist < l_dist[sid]) {
                l_dist[sid] = dist;
                l_idx[sid]  = in_idx[f * nelem + i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned i = get_local_size(1) / 2; i > 0; i >>= 1) {
        if (get_local_id(1) < i) {
            if (valid_feat) {
                To dist = l_dist[sid + i];
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
