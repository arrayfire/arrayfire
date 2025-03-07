/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {

namespace kernel {

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type>
struct dist_op {
    __DH__ To operator()(T v1, T v2) { return v1 - v2; }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SAD> {
    __device__ To operator()(T v1, T v2) {
        return fabsf((float)v1 - (float)v2);
    }
};

template<typename To>
struct dist_op<double, To, AF_SAD> {
    __device__ To operator()(double v1, double v2) {
        return fabs((double)v1 - (double)v2);
    }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SSD> {
    __device__ To operator()(T v1, T v2) { return (v1 - v2) * (v1 - v2); }
};

template<typename To>
struct dist_op<uint, To, AF_SHD> {
    __device__ To operator()(uint v1, uint v2) { return __popc(v1 ^ v2); }
};

template<typename To>
struct dist_op<uintl, To, AF_SHD> {
    __device__ To operator()(uintl v1, uintl v2) { return __popcll(v1 ^ v2); }
};

template<typename To>
struct dist_op<ushort, To, AF_SHD> {
    __device__ To operator()(ushort v1, ushort v2) { return __popc(v1 ^ v2); }
};

template<typename To>
struct dist_op<uchar, To, AF_SHD> {
    __device__ To operator()(uchar v1, uchar v2) { return __popc(v1 ^ v2); }
};

template<typename T, typename To, af_match_type dist_type, bool use_shmem>
__global__ void all_distances(To* out_dist, CParam<T> query, CParam<T> train,
                              const To max_dist, const unsigned feat_len,
                              const unsigned max_feat_len,
                              const unsigned feat_offset) {
    unsigned nquery = query.dims[0];
    unsigned ntrain = train.dims[0];

    unsigned f   = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    __shared__ To s_dist[THREADS];

    extern __shared__ char smem[];
    T* s_query = (T*)smem;
    T* s_train = (T*)smem + max_feat_len;

    s_dist[tid] = max_dist;

    bool valid_feat = (f < ntrain);

    if (valid_feat) {
        // Copy blockDim.x training features to shared memory
        if (use_shmem) {
            unsigned end_feat = min(feat_offset + max_feat_len, feat_len);
            for (unsigned i = feat_offset; i < end_feat; i++) {
                s_train[(i - feat_offset) * blockDim.x + tid] =
                    train.ptr[i * ntrain + f];
            }
        }
    }
    __syncthreads();

    dist_op<T, To, dist_type> op;

    for (unsigned j = 0; j < nquery; j++) {
        s_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < max_feat_len) {
            s_query[tid] = query.ptr[(tid + feat_offset) * nquery + j];
        }
        __syncthreads();

        To dist = 0;
        if (valid_feat) {
            unsigned feat_end = min(feat_offset + max_feat_len, feat_len);
            for (unsigned k = feat_offset; k < feat_end; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
                if (use_shmem) {
                    dist += op(s_train[(k - feat_offset) * blockDim.x + tid],
                               s_query[k - feat_offset]);
                } else {
                    dist +=
                        op(train.ptr[k * ntrain + f], s_query[k - feat_offset]);
                }
            }

            // Only stores the feature index and distance if it's smaller
            // than the best match found so far
            s_dist[tid] = dist;
        }

        __syncthreads();
        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            if (feat_offset == 0)
                out_dist[j * ntrain + f] = s_dist[tid];
            else
                out_dist[j * ntrain + f] += s_dist[tid];
        }
        __syncthreads();
    }
}

template<typename T, typename To, af_match_type dist_type>
void all_distances(Param<To> dist, CParam<T> query, CParam<T> train,
                   const dim_t dist_dim) {
    const dim_t feat_len = query.dims[dist_dim];
    const unsigned max_kern_feat_len =
        std::min(THREADS, static_cast<unsigned>(feat_len));
    const To max_dist = maxval<To>();

    const dim_t sample_dim = (dist_dim == 0) ? 1 : 0;

    const unsigned ntrain = train.dims[sample_dim];

    dim3 threads(THREADS, 1);
    dim3 blocks(divup(ntrain, threads.x), 1);

    // Determine maximum feat_len capable of using shared memory (faster)
    int device          = getActiveDeviceId();
    cudaDeviceProp prop = getDeviceProp(device);
    size_t avail_smem   = prop.sharedMemPerBlock;
    size_t smem_predef =
        2 * THREADS * sizeof(unsigned) + max_kern_feat_len * sizeof(T);
    size_t strain_sz = threads.x * max_kern_feat_len * sizeof(T);
    bool use_shmem   = (avail_smem >= (smem_predef + strain_sz)) ? true : false;
    unsigned smem_sz = (use_shmem) ? smem_predef + strain_sz : smem_predef;

    // For each query vector, find training vector with smallest Hamming
    // distance per CUDA block
    for (dim_t feat_offset = 0; feat_offset < feat_len;
         feat_offset += THREADS) {
        if (use_shmem) {
            CUDA_LAUNCH_SMEM((all_distances<T, To, dist_type, true>), blocks,
                             threads, smem_sz, dist.ptr, query, train, max_dist,
                             feat_len, max_kern_feat_len, feat_offset);
        } else {
            CUDA_LAUNCH_SMEM((all_distances<T, To, dist_type, false>), blocks,
                             threads, smem_sz, dist.ptr, query, train, max_dist,
                             feat_len, max_kern_feat_len, feat_offset);
        }
    }
    POST_LAUNCH_CHECK();
}

}  // namespace kernel

}  // namespace cuda
}  // namespace arrayfire
