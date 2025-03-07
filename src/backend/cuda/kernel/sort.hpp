/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <iota.hpp>
#include <kernel/thrust_sort_by_key.hpp>
#include <math.hpp>
#include <thrust/sort.h>
#include <thrust_utils.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {
// Wrapper functions
template<typename T>
void sort0Iterative(Param<T> val, bool isAscending) {
    for (int w = 0; w < val.dims[3]; w++) {
        int valW = w * val.strides[3];
        for (int z = 0; z < val.dims[2]; z++) {
            int valWZ = valW + z * val.strides[2];
            for (int y = 0; y < val.dims[1]; y++) {
                int valOffset = valWZ + y * val.strides[1];

                if (isAscending) {
                    THRUST_SELECT(thrust::sort, val.ptr + valOffset,
                                  val.ptr + valOffset + val.dims[0]);
                } else {
                    THRUST_SELECT(thrust::sort, val.ptr + valOffset,
                                  val.ptr + valOffset + val.dims[0],
                                  thrust::greater<T>());
                }
            }
        }
    }
    POST_LAUNCH_CHECK();
}

template<typename T>
void sortBatched(Param<T> pVal, int dim, bool isAscending) {
    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pVal.dims[i];

    // Sort dimension
    // tileDims * seqDims = inDims
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    Array<uint> pKey = iota<uint>(seqDims, tileDims);

    pVal = flat(pVal);

    // Sort indices
    // sort_by_key<T, uint, isAscending>(*resVal, *resKey, val, key, 0);
    thrustSortByKey(pVal.ptr, pKey.get(), pVal.dims[0], isAscending);

    // Needs to be ascending (true) in order to maintain the indices properly
    thrustSortByKey(pKey.get(), pVal.ptr, pVal.dims[0], true);
}

template<typename T>
void sort0(Param<T> val, bool isAscending) {
    int higherDims = val.dims[1] * val.dims[2] * val.dims[3];

    if (higherDims > 10)
        sortBatched<T>(val, 0, isAscending);
    else
        kernel::sort0Iterative<T>(val, isAscending);
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
