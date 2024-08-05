/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <type_traits>
#include "config.hpp"

#include <kernel/shfl_intrinsics.hpp>
#include <cub/device/device_reduce.cuh>

using std::unique_ptr;

namespace arrayfire {
namespace cuda {
namespace kernel {

// Reduces keys across block boundaries
template<typename Tk, typename To, af_op_t op>
__global__ void final_boundary_reduce(int *reduced_block_sizes, Param<Tk> keys,
                                      Param<To> vals, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    common::Binary<compute_t<To>, op> reduce;

    if (tid == ((blockIdx.x + 1) * blockDim.x) - 1 &&
        blockIdx.x < gridDim.x - 1) {
        Tk k0 = keys.ptr[tid];
        Tk k1 = keys.ptr[tid + 1];
        if (k0 == k1) {
            compute_t<To> v0                = compute_t<To>(vals.ptr[tid]);
            compute_t<To> v1                = compute_t<To>(vals.ptr[tid + 1]);
            vals.ptr[tid + 1]               = reduce(v0, v1);
            reduced_block_sizes[blockIdx.x] = blockDim.x - 1;
        } else {
            reduced_block_sizes[blockIdx.x] = blockDim.x;
        }
    }

    // if last block, set block size to difference between n and block boundary
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
        reduced_block_sizes[blockIdx.x] = n - (blockIdx.x * blockDim.x);
    }
}

// Tests if data needs further reduction, including across block boundaries
template<typename Tk>
__global__ void test_needs_reduction(int *needs_another_reduction,
                                     int *needs_block_boundary_reduced,
                                     CParam<Tk> keys_in, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tk k;

    if (tid < n) { k = keys_in.ptr[tid]; }

    int update_key = (k == shfl_down_sync(k, 1)) &&
                     (tid < (n - 1)) && ((threadIdx.x % 32) < 31);
    int remaining_updates = any_sync(update_key);

    __syncthreads();

    if (remaining_updates && (threadIdx.x % 32 == 0))
        atomicOr(needs_another_reduction, remaining_updates);

    // check across warp boundaries
    update_key =
        (((threadIdx.x % 32) == 31)           // last thread in warp
         && (threadIdx.x < (blockDim.x - 1))  // not last thread in block
         // next value valid and equal
         && ((tid + 1) < n) && (k == keys_in.ptr[tid + 1]));
    remaining_updates = any_sync(update_key);

    // TODO: single per warp? change to assignment rather than atomicOr
    if (remaining_updates) atomicOr(needs_another_reduction, remaining_updates);

    // last thread in each block checks if any inter-block keys need further
    // reduction
    if (tid == ((blockIdx.x + 1) * blockDim.x) - 1 &&
        blockIdx.x < gridDim.x - 1) {
        int k0 = keys_in.ptr[tid];
        int k1 = keys_in.ptr[tid + 1];
        if (k0 == k1) { atomicOr(needs_block_boundary_reduced, 1); }
    }
}

// Compacts "incomplete" block-sized chunks of data in global memory
template<typename Tk, typename To>
__global__ void compact(int *reduced_block_sizes, Param<Tk> keys_out,
                        Param<To> vals_out, CParam<Tk> keys_in,
                        CParam<To> vals_in, const int nBlocksZ) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z % nBlocksZ;
    const int bidw = blockIdx.z / nBlocksZ;

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite   = (blockIdx.x == 0) ? reduced_block_sizes[0]
                                     : reduced_block_sizes[blockIdx.x] -
                                         reduced_block_sizes[blockIdx.x - 1];
    int writeloc = (blockIdx.x == 0) ? 0 : reduced_block_sizes[blockIdx.x - 1];

    const int bOffset = bidw * vals_in.strides[3] + bidz * vals_in.strides[2] +
                        bidy * vals_in.strides[1];
    Tk k = keys_in.ptr[tidx];
    To v = vals_in.ptr[bOffset + tidx];

    if (threadIdx.x < nwrite) {
        keys_out.ptr[writeloc + threadIdx.x]           = k;
        vals_out.ptr[bOffset + writeloc + threadIdx.x] = v;
    }
}

// Compacts "incomplete" block-sized chunks of data in global memory
template<typename Tk, typename To>
__global__ void compact_dim(int *reduced_block_sizes, Param<Tk> keys_out,
                            Param<To> vals_out, CParam<Tk> keys_in,
                            CParam<To> vals_in, const int dim,
                            const int nBlocksZ) {
    __shared__ int dim_ordering[4];
    if (threadIdx.x == 0) {
        int d           = 1;
        dim_ordering[0] = dim;
        for (int i = 0; i < 4; ++i) {
            if (i != dim) dim_ordering[d++] = i;
        }
    }
    __syncthreads();

    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z % nBlocksZ;
    const int bidw = blockIdx.z / nBlocksZ;

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite   = (blockIdx.x == 0) ? reduced_block_sizes[0]
                                     : reduced_block_sizes[blockIdx.x] -
                                         reduced_block_sizes[blockIdx.x - 1];
    int writeloc = (blockIdx.x == 0) ? 0 : reduced_block_sizes[blockIdx.x - 1];

    const int tid = bidw * vals_in.strides[dim_ordering[3]] +
                    bidz * vals_in.strides[dim_ordering[2]] +
                    bidy * vals_in.strides[dim_ordering[1]] +
                    tidx * vals_in.strides[dim];
    Tk k = keys_in.ptr[tidx];
    To v = vals_in.ptr[tid];

    if (threadIdx.x < nwrite) {
        keys_out.ptr[writeloc + threadIdx.x] = k;
        const int bOffset = bidw * vals_out.strides[dim_ordering[3]] +
                            bidz * vals_out.strides[dim_ordering[2]] +
                            bidy * vals_out.strides[dim_ordering[1]];
        vals_out
            .ptr[bOffset + (writeloc + threadIdx.x) * vals_in.strides[dim]] = v;
    }
}

const static int maxResPerWarp = 32;  // assume dim 0, no NAN values

// Reduces each block by key
template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
__global__ static void reduce_blocks_by_key(int *reduced_block_sizes,
                                            Param<Tk> reduced_keys,
                                            Param<To> reduced_vals,
                                            CParam<Tk> keys, CParam<Ti> vals,
                                            int n, bool change_nan, To nanval,
                                            const int nBlocksZ) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z % nBlocksZ;
    const int bidw = blockIdx.z / nBlocksZ;

    const int laneid = tidx % 32;

    const int nWarps = DIMX / 32;

    //
    // Allocate and initialize shared memory

    __shared__ int
        warpReduceSizes[nWarps];  // number of reduced elements in each warp

    __shared__ compute_t<Tk> warpReduceKeys[nWarps]
                                           [maxResPerWarp];  // reduced key
                                                             // segments for
                                                             // each warp
    __shared__ compute_t<To> warpReduceVals[nWarps]
                                           [maxResPerWarp];  // reduced values
                                                             // for each warp
                                                             // corresponding to
                                                             // each key segment

    // space to hold left/right-most keys of each reduced warp to check if
    // reduction should happen across boundaries
    __shared__ compute_t<Tk> warpReduceLeftBoundaryKeys[nWarps];
    __shared__ compute_t<Tk> warpReduceRightBoundaryKeys[nWarps];

    // space to hold right-most values of each reduced warp to check if
    // reduction should happen across boundaries
    __shared__ compute_t<To> warpReduceRightBoundaryVals[nWarps];

    // space to compact and finalize all reductions within block
    __shared__ compute_t<Tk> warpReduceKeysSmemFinal[nWarps * maxResPerWarp];
    __shared__ compute_t<To> warpReduceValsSmemFinal[nWarps * maxResPerWarp];

    //
    // will hold final number of reduced elements in block
    __shared__ int reducedBlockSize;

    if (threadIdx.x == 0) { reducedBlockSize = 0; }
    if (threadIdx.x < nWarps * maxResPerWarp)
        warpReduceValsSmemFinal[threadIdx.x] = scalar<compute_t<To>>(0);
    __syncthreads();

    common::Binary<compute_t<To>, op> reduce;
    common::Transform<compute_t<Ti>, compute_t<To>, op> transform;

    // load keys and values to threads
    compute_t<Tk> k;
    compute_t<To> v;
    if (tidx < n) {
        const int tid = bidw * vals.strides[3] + bidz * vals.strides[2] +
                        bidy * vals.strides[1] +
                        tidx;  // index for batched inputs
        k = keys.ptr[tidx];
        v = transform(compute_t<Ti>(vals.ptr[tid]));
        if (change_nan) v = IS_NAN(v) ? compute_t<To>(nanval) : v;
    } else {
        v = common::Binary<compute_t<To>, op>::init();
    }

    compute_t<Tk> eq_check = (k != shfl_up_sync(k, 1));
    // mark threads containing unique keys
    char unique_flag = (eq_check || (laneid == 0)) && (tidx < n);

    // scan unique flags to enumerate unique keys
    char unique_id = unique_flag;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        char y = shfl_up_sync(unique_id, offset);
        if (laneid >= offset) unique_id += y;
    }

    //
    // Reduce each warp by key
    char all_eq = (k == shfl_down_sync(k, 1));
    if (all_sync(all_eq)) {  // check special case of single key per warp
        v = reduce(v, shfl_down_sync(v, 1));
        v = reduce(v, shfl_down_sync(v, 2));
        v = reduce(v, shfl_down_sync(v, 4));
        v = reduce(v, shfl_down_sync(v, 8));
        v = reduce(v, shfl_down_sync(v, 16));
    } else {
        compute_t<To> init = common::Binary<compute_t<To>, op>::init();
        int eq_check, update_key;
#pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            eq_check =
                (unique_id == shfl_down_sync(unique_id, delta));

            // checks if this thread should perform a reduction
            update_key =
                eq_check && (laneid < (32 - delta)) && ((tidx + delta) < n);

            // shfls data from neighboring threads
            compute_t<To> uval = shfl_down_sync(v, delta);

            // update if thread requires it
            v = reduce(v, (update_key ? uval : init));
        }
    }

    const int warpid = threadIdx.x / 32;

    // last thread in warp has reduced warp size due to scan^
    if (laneid == 31) { warpReduceSizes[warpid] = unique_id; }

    // write left boundary values for each warp
    if (unique_flag && unique_id == 1) {
        warpReduceLeftBoundaryKeys[warpid] = k;
    }

    // write right boundary values for each warp
    if (unique_flag && unique_id == warpReduceSizes[warpid]) {
        warpReduceRightBoundaryKeys[warpid] = k;
        warpReduceRightBoundaryVals[warpid] = v;
    }

    __syncthreads();

    // if rightmost thread, check next warp's kv,
    // invalidate self and change warpReduceSizes since first thread of next
    // warp will update same key
    // TODO: what if extra empty warps???
    if (unique_flag && unique_id == warpReduceSizes[warpid] &&
        warpid < nWarps - 1) {
        int tid_next_warp = (blockIdx.x * blockDim.x + (warpid + 1) * 32);
        // check within data range
        if (tid_next_warp < n && k == warpReduceLeftBoundaryKeys[warpid + 1]) {
            // disable writing from warps that need carry but aren't terminal
            if (warpReduceSizes[warpid] > 1 || warpid > 0) { unique_flag = 0; }
        }
    }
    __syncthreads();

    // if leftmost thread, reduce carryover from previous warp(s) if needed
    if (unique_flag && unique_id == 1 && warpid > 0) {
        int test_wid = warpid - 1;
        while (test_wid >= 0 && k == warpReduceRightBoundaryKeys[test_wid]) {
            v = reduce(v, warpReduceRightBoundaryVals[test_wid]);
            --warpReduceSizes[test_wid];
            if (warpReduceSizes[test_wid] > 1) break;

            --test_wid;
        }
    }

    if (unique_flag) {
        warpReduceKeys[warpid][unique_id - 1] = k;
        warpReduceVals[warpid][unique_id - 1] = v;
    }

    __syncthreads();

    // at this point, we have nWarps lists in shared memory with each list's
    // size located in the warpReduceSizes[] array
    // perform warp-scan to determine each warp's write location
    int warpSzScan = 0;
    if (warpid == 0 && laneid < nWarps) {
        warpSzScan     = warpReduceSizes[laneid];
        int activemask = 0xFFFFFFFF >> (32 - nWarps);
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            char y = __shfl_up_sync(activemask, warpSzScan, offset);
            if (laneid >= offset) warpSzScan += y;
        }
        warpReduceSizes[laneid] = warpSzScan;
        // final thread has final reduced size of block
        if (laneid == nWarps - 1) reducedBlockSize = warpSzScan;
    }
    __syncthreads();

    // write reduced block size to global memory
    if (threadIdx.x == 0) {
        reduced_block_sizes[blockIdx.x] = reducedBlockSize;
    }

    // compact reduced keys and values before writing to global memory
    if (warpid > 0) {
        int wsz = warpReduceSizes[warpid] - warpReduceSizes[warpid - 1];
        if (laneid < wsz) {
            int warpOffset = warpReduceSizes[warpid - 1];
            warpReduceKeysSmemFinal[warpOffset + laneid] =
                warpReduceKeys[warpid][laneid];
            warpReduceValsSmemFinal[warpOffset + laneid] =
                warpReduceVals[warpid][laneid];
        }
    } else {
        int wsz = warpReduceSizes[warpid];
        if (laneid < wsz) {
            warpReduceKeysSmemFinal[laneid] = warpReduceKeys[0][laneid];
            warpReduceValsSmemFinal[laneid] = warpReduceVals[0][laneid];
        }
    }
    __syncthreads();

    const int bOffset = bidw * reduced_vals.strides[3] +
                        bidz * reduced_vals.strides[2] +
                        bidy * reduced_vals.strides[1];
    // write reduced keys/values per-block
    if (threadIdx.x < reducedBlockSize) {
        reduced_keys.ptr[(blockIdx.x * blockDim.x) + threadIdx.x] =
            warpReduceKeysSmemFinal[threadIdx.x];
        reduced_vals.ptr[bOffset + (blockIdx.x * blockDim.x) + threadIdx.x] =
            warpReduceValsSmemFinal[threadIdx.x];
    }
}

// Reduces each block by key
template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
__global__ static void reduce_blocks_dim_by_key(
    int *reduced_block_sizes, Param<Tk> reduced_keys, Param<To> reduced_vals,
    CParam<Tk> keys, CParam<Ti> vals, int n, bool change_nan, To nanval,
    int dim, const int nBlocksZ) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z % nBlocksZ;
    const int bidw = blockIdx.z / nBlocksZ;

    const int laneid = tidx % 32;
    const int nWarps = DIMX / 32;

    //
    // Allocate and initialize shared memory

    __shared__ int
        warpReduceSizes[nWarps];  // number of reduced elements in each warp

    __shared__ Tk warpReduceKeys[nWarps][maxResPerWarp];  // reduced key
                                                          // segments for each
                                                          // warp
    __shared__ compute_t<To> warpReduceVals[nWarps]
                                           [maxResPerWarp];  // reduced values
                                                             // for each warp
                                                             // corresponding to
                                                             // each key segment

    // space to hold left/right-most keys of each reduced warp to check if
    // reduction should happen accros boundaries
    __shared__ Tk warpReduceLeftBoundaryKeys[nWarps];
    __shared__ Tk warpReduceRightBoundaryKeys[nWarps];

    // space to hold right-most values of each reduced warp to check if
    // reduction should happen accros boundaries
    __shared__ compute_t<To> warpReduceRightBoundaryVals[nWarps];

    // space to compact and finalize all reductions within block
    __shared__ Tk warpReduceKeysSmemFinal[nWarps * maxResPerWarp];
    __shared__ compute_t<To> warpReduceValsSmemFinal[nWarps * maxResPerWarp];

    //
    // will hold final number of reduced elements in block
    __shared__ int reducedBlockSize;
    __shared__ int dim_ordering[4];

    compute_t<To> init = common::Binary<compute_t<To>, op>::init();

    if (threadIdx.x == 0) {
        reducedBlockSize = 0;
        int d            = 1;
        dim_ordering[0]  = dim;
        for (int i = 0; i < 4; ++i) {
            if (i != dim) dim_ordering[d++] = i;
        }
    }
    if (threadIdx.x < nWarps * maxResPerWarp)
        warpReduceValsSmemFinal[threadIdx.x] = init;
    __syncthreads();

    common::Binary<compute_t<To>, op> reduce;
    common::Transform<compute_t<Ti>, compute_t<To>, op> transform;

    // load keys and values to threads
    Tk k;
    compute_t<To> v;
    if (tidx < n) {
        const int tid = bidw * vals.strides[dim_ordering[3]] +
                        bidz * vals.strides[dim_ordering[2]] +
                        bidy * vals.strides[dim_ordering[1]] +
                        tidx * vals.strides[dim];  // index for batched inputs

        k = keys.ptr[tidx];
        v = transform(compute_t<Ti>(vals.ptr[tid]));
        if (change_nan) v = IS_NAN(v) ? compute_t<To>(nanval) : v;
    } else {
        v = init;
    }

    Tk eq_check = (k != shfl_up_sync(k, 1));
    // mark threads containing unique keys
    char unique_flag = (eq_check || (laneid == 0)) && (tidx < n);

    // scan unique flags to enumerate unique keys
    char unique_id = unique_flag;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        char y = shfl_up_sync(unique_id, offset);
        if (laneid >= offset) unique_id += y;
    }

    //
    // Reduce each warp by key
    char all_eq = (k == shfl_down_sync(k, 1));
    if (all_sync(all_eq)) {  // check special case of single key per warp
        v = reduce(v, shfl_down_sync(v, 1));
        v = reduce(v, shfl_down_sync(v, 2));
        v = reduce(v, shfl_down_sync(v, 4));
        v = reduce(v, shfl_down_sync(v, 8));
        v = reduce(v, shfl_down_sync(v, 16));
    } else {
        compute_t<To> init = common::Binary<compute_t<To>, op>::init();
        int eq_check, update_key;
#pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            eq_check =
                (unique_id == shfl_down_sync(unique_id, delta));

            // checks if this thread should perform a reduction
            update_key =
                eq_check && (laneid < (32 - delta)) && ((tidx + delta) < n);

            // shfls data from neighboring threads
            compute_t<To> uval = shfl_down_sync(v, delta);

            // update if thread requires it
            v = reduce(v, (update_key ? uval : init));
        }
    }

    const int warpid = threadIdx.x / 32;

    // last thread in warp has reduced warp size due to scan^
    if (laneid == 31) { warpReduceSizes[warpid] = unique_id; }

    // write left boundary values for each warp
    if (unique_flag && unique_id == 1) {
        warpReduceLeftBoundaryKeys[warpid] = k;
    }

    // write right boundary values for each warp
    if (unique_flag && unique_id == warpReduceSizes[warpid]) {
        warpReduceRightBoundaryKeys[warpid] = k;
        warpReduceRightBoundaryVals[warpid] = v;
    }

    __syncthreads();

    // if rightmost thread, check next warp's kv,
    // invalidate self and change warpReduceSizes since first thread of next
    // warp will update same key
    // TODO: what if extra empty warps???
    if (unique_flag && unique_id == warpReduceSizes[warpid] &&
        warpid < nWarps - 1) {
        int tid_next_warp = (blockIdx.x * blockDim.x + (warpid + 1) * 32);
        // check within data range
        if (tid_next_warp < n && k == warpReduceLeftBoundaryKeys[warpid + 1]) {
            // disable writing from warps that need carry but aren't terminal
            if (warpReduceSizes[warpid] > 1 || warpid > 0) { unique_flag = 0; }
        }
    }
    __syncthreads();

    // if leftmost thread, reduce carryover from previous warp(s) if needed
    if (unique_flag && unique_id == 1 && warpid > 0) {
        int test_wid = warpid - 1;
        while (test_wid >= 0 && k == warpReduceRightBoundaryKeys[test_wid]) {
            v = reduce(v, warpReduceRightBoundaryVals[test_wid]);
            --warpReduceSizes[test_wid];
            if (warpReduceSizes[test_wid] > 1) break;

            --test_wid;
        }
    }

    if (unique_flag) {
        warpReduceKeys[warpid][unique_id - 1] = k;
        warpReduceVals[warpid][unique_id - 1] = v;
    }

    __syncthreads();

    // at this point, we have nWarps lists in shared memory with each list's
    // size located in the warpReduceSizes[] array
    // perform warp-scan to determine each warp's write location
    int warpSzScan = 0;
    if (warpid == 0 && laneid < nWarps) {
        warpSzScan     = warpReduceSizes[laneid];
        int activemask = 0xFFFFFFFF >> (32 - nWarps);
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            char y = __shfl_up_sync(activemask, warpSzScan, offset);
            if (laneid >= offset) warpSzScan += y;
        }
        warpReduceSizes[laneid] = warpSzScan;
        // final thread has final reduced size of block
        if (laneid == nWarps - 1) reducedBlockSize = warpSzScan;
    }
    __syncthreads();

    // write reduced block size to global memory
    if (threadIdx.x == 0) {
        reduced_block_sizes[blockIdx.x] = reducedBlockSize;
    }

    // compact reduced keys and values before writing to global memory
    if (warpid > 0) {
        int wsz = warpReduceSizes[warpid] - warpReduceSizes[warpid - 1];
        if (laneid < wsz) {
            int warpOffset = warpReduceSizes[warpid - 1];
            warpReduceKeysSmemFinal[warpOffset + laneid] =
                warpReduceKeys[warpid][laneid];
            warpReduceValsSmemFinal[warpOffset + laneid] =
                warpReduceVals[warpid][laneid];
        }
    } else {
        int wsz = warpReduceSizes[warpid];
        if (laneid < wsz) {
            warpReduceKeysSmemFinal[laneid] = warpReduceKeys[0][laneid];
            warpReduceValsSmemFinal[laneid] = warpReduceVals[0][laneid];
        }
    }
    __syncthreads();

    // write reduced keys/values per-block
    if (threadIdx.x < reducedBlockSize) {
        const int bOffset = bidw * reduced_vals.strides[dim_ordering[3]] +
                            bidz * reduced_vals.strides[dim_ordering[2]] +
                            bidy * reduced_vals.strides[dim_ordering[1]];
        reduced_keys.ptr[(blockIdx.x * blockDim.x) + threadIdx.x] =
            warpReduceKeysSmemFinal[threadIdx.x];
        reduced_vals.ptr[bOffset + ((blockIdx.x * blockDim.x) + threadIdx.x) *
                                       reduced_vals.strides[dim]] =
            warpReduceValsSmemFinal[threadIdx.x];
    }
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
