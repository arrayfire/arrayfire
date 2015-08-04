/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <backend.hpp>

namespace cuda
{

namespace kernel
{

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type>
struct dist_op
{
    __DH__ To operator()(T v1, T v2)
    {
        return v1 - v2;
    }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SAD>
{
    __device__ To operator()(T v1, T v2)
    {
        return abs((double)v1 - (double)v2);
    }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SSD>
{
    __device__ To operator()(T v1, T v2)
    {
        return (v1 - v2) * (v1 - v2);
    }
};

template<typename To>
struct dist_op<uint, To, AF_SHD>
{
    __device__ To operator()(uint v1, uint v2)
    {
        return __popc(v1 ^ v2);
    }
};

template<typename To>
struct dist_op<uintl, To, AF_SHD>
{
    __device__ To operator()(uintl v1, uintl v2)
    {
        return __popc(v1 ^ v2);
    }
};

template<typename To>
struct dist_op<uchar, To, AF_SHD>
{
    __device__ To operator()(uchar v1, uchar v2)
    {
        return __popc(v1 ^ v2);
    }
};


template<typename T, typename To, af_match_type dist_type, unsigned feat_len, bool use_shmem>
__global__ void nearest_neighbour_unroll(
    unsigned* out_idx,
    To* out_dist,
    CParam<T> query,
    CParam<T> train,
    const To max_dist)
{
    unsigned nquery = query.dims[0];
    unsigned ntrain = train.dims[0];

    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    __shared__ To       s_dist[THREADS];
    __shared__ unsigned s_idx[THREADS];

    extern __shared__ char smem[];
    T* s_query = (T*)smem;
    T* s_train = (T*)smem + feat_len;

    s_dist[tid] = max_dist;
    s_idx[tid]  = 0xffffffff;

    bool valid_feat = (f < ntrain);

    if (valid_feat) {
        // Copy blockDim.x training features to shared memory
        if (use_shmem) {
            #pragma unroll
            for (unsigned i = 0; i < feat_len; i++) {
                s_train[i * blockDim.x + tid] = train.ptr[i * ntrain + f];
            }
        }
    }
    __syncthreads();

    dist_op<T, To, dist_type> op;

    for (unsigned j = 0; j < nquery; j++) {
        s_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < feat_len && valid_feat) {
            s_query[tid] = query.ptr[tid * nquery + j];
        }
        __syncthreads();

        To dist = 0;
        if (valid_feat) {
            #pragma unroll
            for (unsigned k = 0; k < feat_len; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
                if (use_shmem) {
                    dist += op(s_train[k * blockDim.x + tid], s_query[k]);
                }
                else {
                    dist += op(train.ptr[k * ntrain + f], s_query[k]);
                }
            }

            // Only stores the feature index and distance if it's smaller
            // than the best match found so far
            s_dist[tid] = dist;
            s_idx[tid]  = f;
        }
        __syncthreads();

        // Find best match in training features from block to the current
        // query feature
        if (tid < 128) {
            if (s_dist[tid + 128] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 128];
                s_idx[tid]  = s_idx[tid + 128];
            }
        }
        __syncthreads();
        if (tid < 64) {
            if (s_dist[tid + 64] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 64];
                s_idx[tid]  = s_idx[tid + 64];
            }
        }
        __syncthreads();
        if (tid < 32) {
            if (s_dist[tid + 32] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 32];
                s_idx[tid]  = s_idx[tid + 32];
            }
        }
        __syncthreads();
        if (tid < 16) {
            if (s_dist[tid + 16] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 16];
                s_idx[tid]  = s_idx[tid + 16];
            }
        }
        __syncthreads();
        if (tid < 8) {
            if (s_dist[tid + 8] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 8];
                s_idx[tid]  = s_idx[tid + 8];
            }
        }
        __syncthreads();
        if (tid < 4) {
            if (s_dist[tid + 4] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 4];
                s_idx[tid]  = s_idx[tid + 4];
            }
        }
        __syncthreads();
        if (tid < 2) {
            if (s_dist[tid + 2] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 2];
                s_idx[tid]  = s_idx[tid + 2];
            }
        }
        __syncthreads();
        if (tid < 1) {
            if (s_dist[tid + 1] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 1];
                s_idx[tid]  = s_idx[tid + 1];
            }
        }
        __syncthreads();

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * gridDim.x + blockIdx.x] = s_dist[0];
            out_idx[j * gridDim.x + blockIdx.x]  = s_idx[0];
        }
        __syncthreads();
    }
}

template<typename T, typename To, af_match_type dist_type, bool use_shmem>
__global__ void nearest_neighbour(
    unsigned* out_idx,
    To* out_dist,
    CParam<T> query,
    CParam<T> train,
    const To max_dist,
    const unsigned feat_len)
{
    unsigned nquery = query.dims[0];
    unsigned ntrain = train.dims[0];

    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    __shared__ To s_dist[THREADS];
    __shared__ unsigned s_idx[THREADS];

    extern __shared__ char smem[];
    T* s_query = (T*)smem;
    T* s_train = (T*)smem + feat_len;

    s_dist[tid] = max_dist;
    s_idx[tid]  = 0xffffffff;

    bool valid_feat = (f < ntrain);

    if (valid_feat) {
        // Copy blockDim.x training features to shared memory
        if (use_shmem) {
            for (unsigned i = 0; i < feat_len; i++) {
                s_train[i * blockDim.x + tid] = train.ptr[i * ntrain + f];
            }
        }
    }
    __syncthreads();

    dist_op<T, To, dist_type> op;

    for (unsigned j = 0; j < nquery; j++) {
        s_dist[tid] = max_dist;

        // Load one query feature that will be tested against all training
        // features in current block
        if (tid < feat_len && valid_feat) {
            s_query[tid] = query.ptr[tid * nquery + j];
        }
        __syncthreads();

        To dist = 0;
        if (valid_feat) {
            for (unsigned k = 0; k < feat_len; k++) {
                // Calculate Hamming distance for 32-bits of descriptor and
                // accumulates to dist
                if (use_shmem) {
                    dist += op(s_train[k * blockDim.x + tid], s_query[k]);
                }
                else {
                    dist += op(train.ptr[k * ntrain + f], s_query[k]);
                }
            }

            // Only stores the feature index and distance if it's smaller
            // than the best match found so far
            s_dist[tid] = dist;
            s_idx[tid]  = f;
        }
        __syncthreads();

        // Find best match in training features from block to the current
        // query feature
        if (tid < 128) {
            if (s_dist[tid + 128] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 128];
                s_idx[tid]  = s_idx[tid + 128];
            }
        }
        __syncthreads();
        if (tid < 64) {
            if (s_dist[tid + 64] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 64];
                s_idx[tid]  = s_idx[tid + 64];
            }
        }
        __syncthreads();
        if (tid < 32) {
            if (s_dist[tid + 32] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 32];
                s_idx[tid]  = s_idx[tid + 32];
            }
        }
        __syncthreads();
        if (tid < 16) {
            if (s_dist[tid + 16] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 16];
                s_idx[tid]  = s_idx[tid + 16];
            }
        }
        __syncthreads();
        if (tid < 8) {
            if (s_dist[tid + 8] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 8];
                s_idx[tid]  = s_idx[tid + 8];
            }
        }
        __syncthreads();
        if (tid < 4) {
            if (s_dist[tid + 4] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 4];
                s_idx[tid]  = s_idx[tid + 4];
            }
        }
        __syncthreads();
        if (tid < 2) {
            if (s_dist[tid + 2] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 2];
                s_idx[tid]  = s_idx[tid + 2];
            }
        }
        __syncthreads();
        if (tid < 1) {
            if (s_dist[tid + 1] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + 1];
                s_idx[tid]  = s_idx[tid + 1];
            }
        }
        __syncthreads();

        // Store best match in training features from block to the current
        // query feature
        if (valid_feat) {
            out_dist[j * gridDim.x + blockIdx.x] = s_dist[0];
            out_idx[j * gridDim.x + blockIdx.x]  = s_idx[0];
        }
        __syncthreads();
    }
}

template<typename To>
__global__ void select_matches(
    Param<unsigned> idx,
    Param<To> dist,
    const unsigned* in_idx,
    const To* in_dist,
    const unsigned nfeat,
    const unsigned nelem,
    const To max_dist)
{
    unsigned f = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned sid = threadIdx.x * blockDim.y + threadIdx.y;

    __shared__ To s_dist[THREADS];
    __shared__ unsigned s_idx[THREADS];

    if (f < nfeat) {
        s_dist[sid] = max_dist;
        __syncthreads();

        for (unsigned i = threadIdx.y; i < nelem; i += blockDim.y) {
            To dist = in_dist[f * nelem + i];

            // Copy all best matches previously found in nearest_neighbour() to
            // shared memory
            if (dist < s_dist[sid]) {
                s_dist[sid] = dist;
                s_idx[sid]  = in_idx[f * nelem + i];
            }
            __syncthreads();
        }

        // Reduce best matches and find the best of them all
        for (unsigned i = blockDim.y / 2; i > 0; i >>= 1) {
            if (threadIdx.y < i) {
                To dist = s_dist[sid + i];
                if (dist < s_dist[sid]) {
                    s_dist[sid] = dist;
                    s_idx[sid]  = s_idx[sid + i];
                }
                __syncthreads();
            }
        }

        // Store best matches and indexes to training dataset
        if (threadIdx.y == 0) {
            dist.ptr[f] = s_dist[threadIdx.x * blockDim.y];
            idx.ptr[f]  = s_idx[threadIdx.x * blockDim.y];
        }
    }
}

template<typename T, typename To, af_match_type dist_type>
void nearest_neighbour(Param<uint> idx,
                       Param<To> dist,
                       CParam<T> query,
                       CParam<T> train,
                       const dim_t dist_dim,
                       const unsigned n_dist)
{
    const unsigned feat_len = query.dims[dist_dim];
    const To max_dist = limit_max<To>();

    if (feat_len > THREADS) {
        CUDA_NOT_SUPPORTED();
    }

    const dim_t sample_dim = (dist_dim == 0) ? 1 : 0;

    const unsigned nquery = query.dims[sample_dim];
    const unsigned ntrain = train.dims[sample_dim];

    dim3 threads(THREADS, 1);
    dim3 blocks(divup(ntrain, threads.x), 1);

    // Determine maximum feat_len capable of using shared memory (faster)
    int device = getActiveDeviceId();
    cudaDeviceProp prop = getDeviceProp(device);
    size_t avail_smem = prop.sharedMemPerBlock;
    size_t smem_predef = 2 * THREADS * sizeof(unsigned) + feat_len * sizeof(T);
    size_t strain_sz = threads.x * feat_len * sizeof(T);
    bool use_shmem = (avail_smem >= (smem_predef + strain_sz)) ? true : false;
    unsigned smem_sz = (use_shmem) ? smem_predef + strain_sz : smem_predef;

    unsigned nblk = blocks.x;

    unsigned* d_blk_idx  = memAlloc<unsigned>(nblk * nquery);
    To* d_blk_dist = memAlloc<To>(nblk * nquery);

    // For each query vector, find training vector with smallest Hamming
    // distance per CUDA block
    if (use_shmem) {
        switch(feat_len) {
        // Optimized lengths (faster due to loop unrolling)
        case 1:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,1,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 2:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,2,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 4:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,4,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 8:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,8,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 16:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,16,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 32:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,32,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 64:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,64,true>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        default:
            CUDA_LAUNCH_SMEM((nearest_neighbour<T,To,dist_type,true>), blocks, threads, smem_sz,
                           d_blk_idx, d_blk_dist, query, train, max_dist, feat_len);
        }
    }
    else {
        switch(feat_len) {
        // Optimized lengths (faster due to loop unrolling)
        case 1:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,1,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 2:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,2,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 4:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,4,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 8:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,8,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 16:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,16,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 32:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,32,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        case 64:
            CUDA_LAUNCH_SMEM((nearest_neighbour_unroll<T,To,dist_type,64,false>), blocks, threads, smem_sz,
                                  d_blk_idx, d_blk_dist, query, train, max_dist);
            break;
        default:
            CUDA_LAUNCH_SMEM((nearest_neighbour<T,To,dist_type,false>), blocks, threads, smem_sz,
                           d_blk_idx, d_blk_dist, query, train, max_dist, feat_len);
        }
    }
    POST_LAUNCH_CHECK();

    threads = dim3(32, 8);
    blocks = dim3(nquery, 1);

    // Reduce all smallest Hamming distances from each block and store final
    // best match
    CUDA_LAUNCH(select_matches, blocks, threads,
            idx, dist, d_blk_idx, d_blk_dist, nquery, nblk, max_dist);
    POST_LAUNCH_CHECK();

    memFree(d_blk_idx);
    memFree(d_blk_dist);
}

} // namespace kernel

} // namespace cuda
