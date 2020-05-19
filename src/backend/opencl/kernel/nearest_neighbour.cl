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
__inline unsigned popcount(unsigned x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x0000003F;
}
#endif

#ifdef USE_DOUBLE
To _sad_(T v1, T v2) { return fabs(v1 - v2); }
#else
To _sad_(T v1, T v2) { return fabs((float)v1 - (float)v2); }
#endif

To _ssd_(T v1, T v2) { return (v1 - v2) * (v1 - v2); }

#ifdef __SHD__
unsigned _shd_(T v1, T v2) { return popcount(v1 ^ v2); }
#endif

kernel void knnAllDistances(global To* out_dist, global const T* query,
                            KParam qInfo, global const T* train, KParam tInfo,
                            const To max_dist, const unsigned feat_len,
                            const unsigned max_feat_len,
                            const unsigned feat_offset, local T* lmem) {
    unsigned nquery = qInfo.dims[0];
    unsigned ntrain = tInfo.dims[0];

    unsigned f   = get_global_id(0);
    unsigned tid = get_local_id(0);

    local To l_dist[THREADS];

    local T* l_query = lmem;
    local T* l_train = lmem + max_feat_len;

    l_dist[tid] = max_dist;

    bool valid_feat = (f < ntrain);

#ifdef USE_LOCAL_MEM
    if (valid_feat) {
        // Copy local_size(0) training features to shared memory
        unsigned end_feat = min(feat_offset + max_feat_len, feat_len);
        for (unsigned i = feat_offset; i < feat_len; i++) {
            l_train[(i - feat_offset) * get_local_size(0) + tid] =
                train[i * ntrain + f + tInfo.offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int j = 0; j < (int)nquery; j++) {
        l_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < max_feat_len) {
            l_query[tid] =
                query[(tid + feat_offset) * nquery + j + qInfo.offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        To dist = 0;
        if (valid_feat) {
            unsigned feat_end = min(feat_offset + max_feat_len, feat_len);
            for (unsigned k = feat_offset; k < feat_end; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
#ifdef USE_LOCAL_MEM
                dist +=
                    DISTOP(l_train[(k - feat_offset) * get_local_size(0) + tid],
                           l_query[k - feat_offset]);
#else
                dist += DISTOP(train[k * ntrain + f + tInfo.offset],
                               l_query[k - feat_offset]);
#endif
            }
        }

        // Only stores the feature index and distance if it's smaller
        // than the best match found so far
        if (valid_feat) { l_dist[tid] = dist; }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            if (feat_offset == 0)
                out_dist[j * ntrain + f] = l_dist[tid];
            else
                out_dist[j * ntrain + f] += l_dist[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
