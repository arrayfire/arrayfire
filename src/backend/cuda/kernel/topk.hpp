/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cub/block/block_radix_sort.cuh>

#include <Array.hpp>
#include <Param.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include <types.hpp>

#include <limits>

using cub::BlockRadixSort;

namespace arrayfire {
namespace cuda {
namespace kernel {
static const int TOPK_THRDS_PER_BLK = 256;
static const int TOPK_IDX_THRD_LOAD = 4;

template<typename T, bool READ_INDEX>
static __global__ void kerTopkDim0(Param<T> ovals, Param<uint> oidxs,
                                   CParam<T> ivals, CParam<uint> iidxs,
                                   const int k, const af::topkFunction order,
                                   uint numLaunchBlocksY) {
    using ValueType       = uint;
    using BlockRadixSortT = BlockRadixSort<compute_t<T>, TOPK_THRDS_PER_BLK,
                                           TOPK_IDX_THRD_LOAD, ValueType>;

    __shared__ typename BlockRadixSortT::TempStorage smem;

    const int bw = blockIdx.y / numLaunchBlocksY;
    const int bz = blockIdx.z;
    const int by = (blockIdx.y - bw * numLaunchBlocksY);

    const uint gx       = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gxStride = blockDim.x * gridDim.x;
    const uint elements = ivals.dims[0];

    const data_t<T>* kdata = ivals.ptr + by * ivals.strides[1] +
                             bz * ivals.strides[2] + bw * ivals.strides[3];

    const ValueType* idata = iidxs.ptr + by * iidxs.strides[1] +
                             bz * iidxs.strides[2] + bw * iidxs.strides[3];

    T* ores = ovals.ptr + by * ovals.strides[1] + bz * ovals.strides[2] +
              bw * ovals.strides[3];
    uint* ires = oidxs.ptr + by * oidxs.strides[1] + bz * oidxs.strides[2] +
                 bw * oidxs.strides[3];

    compute_t<T> keys[TOPK_IDX_THRD_LOAD];
    ValueType vals[TOPK_IDX_THRD_LOAD];

    for (uint li = 0, i = gx; li < TOPK_IDX_THRD_LOAD; i += gxStride, li++) {
        if (i < elements) {
            keys[li] = static_cast<compute_t<T>>(kdata[i]);
            vals[li] = (READ_INDEX) ? idata[i] : i;
        } else {
            keys[li] = (order == AF_TOPK_MAX) ? minval<compute_t<T>>()
                                              : maxval<compute_t<T>>();
            vals[li] = maxval<ValueType>();
        }
    }

    if (order == AF_TOPK_MAX) {
        BlockRadixSortT(smem).SortDescendingBlockedToStriped(keys, vals);
    } else {
        BlockRadixSortT(smem).SortBlockedToStriped(keys, vals);
    }

    if (threadIdx.x < k) {
        int oidx   = threadIdx.x + blockIdx.x * k;
        ores[oidx] = keys[0];
        ires[oidx] = vals[0];
    }
}

template<typename T>
void topkDim0(Param<T> ovals, Param<uint> oidxs, CParam<T> ivals, const int k,
              const af::topkFunction order) {
    dim3 threads(TOPK_THRDS_PER_BLK, 1);
    const int thrdLoad = TOPK_IDX_THRD_LOAD;

    int numBlocksX = divup(ivals.dims[0], threads.x * thrdLoad);
    dim3 blocks(numBlocksX, ivals.dims[1] * ivals.dims[3], ivals.dims[2]);

    // The algorithm is to iteratively find top k elements among each block
    // of threads until there is only one block to launch.
    // The additional memory used for values and indices is allocated only
    // before the first iteration and reused for further iterations.

    // Temporary storage allocation for iterations
    Array<T> tvals    = createEmptyArray<T>(dim4());
    Array<uint> tidxs = createEmptyArray<uint>(dim4());

    if (numBlocksX > 1) {
        tvals = createEmptyArray<T>(dim4(k * numBlocksX, ivals.dims[1]));
        // TODO(umar): this can be smaller because the first iteration is not
        // reading this array.
        tidxs = createEmptyArray<uint>(dim4(k * numBlocksX, ivals.dims[1]));
    }

    int prevBlocksX = 1;

    CParam<T> iivals    = ivals;
    CParam<uint> iiidxs = tidxs;

    int dims0      = tvals.dims()[0];
    bool first_run = true;
    do {
        if (blocks.x == 1) {
            tvals = createParamArray(ovals, false);
            tidxs = createParamArray(oidxs, false);
        }

        if (first_run) {
            // Launch topk which doesn't read the indice values from global
            // memory
            CUDA_LAUNCH((kerTopkDim0<T, false>), blocks, threads, tvals, tidxs,
                        iivals, iiidxs, k, order, ivals.dims[1]);
            first_run = false;
        } else {
            CUDA_LAUNCH((kerTopkDim0<T, true>), blocks, threads, tvals, tidxs,
                        iivals, iiidxs, k, order, ivals.dims[1]);
        }

        POST_LAUNCH_CHECK();

        prevBlocksX = blocks.x;
        blocks.x    = divup(dims0, threads.x * thrdLoad);

        // set output of current iteration as input for the next iteration
        iivals = tvals;
        iiidxs = tidxs;

        dims0 = blocks.x * k;

        tvals.setDataDims(dim4(dims0, tvals.elements() / (float)dims0));
        tidxs.setDataDims(dim4(dims0, tidxs.elements() / (float)dims0));
    } while (prevBlocksX > 1);
}

template<typename T>
inline void topk(Param<T> ovals, Param<uint> oidxs, CParam<T> ivals,
                 const int k, const int dim, const af::topkFunction order) {
    assert(dim == 0);
    // TODO Add switch statement when support for other dims is added
    topkDim0<T>(ovals, oidxs, ivals, k, order);
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
