/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <iota.hpp>
#include <kernel/sort_helper.hpp>
#include <traits.hpp>

AF_DEPRECATED_WARNINGS_OFF
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
AF_DEPRECATED_WARNINGS_ON

namespace compute = boost::compute;

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void sort0Iterative(Param val, bool isAscending) {
    compute::command_queue c_queue(getQueue()());

    compute::buffer val_buf((*val.data)());

    for (int w = 0; w < val.info.dims[3]; w++) {
        int valW = w * val.info.strides[3];
        for (int z = 0; z < val.info.dims[2]; z++) {
            int valWZ = valW + z * val.info.strides[2];
            for (int y = 0; y < val.info.dims[1]; y++) {
                int valOffset = valWZ + y * val.info.strides[1];

                if (isAscending) {
                    compute::sort(compute::make_buffer_iterator<type_t<T>>(
                                      val_buf, valOffset),
                                  compute::make_buffer_iterator<type_t<T>>(
                                      val_buf, valOffset + val.info.dims[0]),
                                  compute::less<type_t<T>>(), c_queue);
                } else {
                    compute::sort(compute::make_buffer_iterator<type_t<T>>(
                                      val_buf, valOffset),
                                  compute::make_buffer_iterator<type_t<T>>(
                                      val_buf, valOffset + val.info.dims[0]),
                                  compute::greater<type_t<T>>(), c_queue);
                }
            }
        }
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void sortBatched(Param pVal, int dim, bool isAscending) {
    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pVal.info.dims[i];

    // Sort dimension
    // tileDims * seqDims = inDims
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    // Array<uint> pKey = createEmptyArray<uint>(inDims);
    Array<uint> pKey = iota<uint>(seqDims, tileDims);

    pKey.setDataDims(inDims.elements());

    // Flat
    pVal.info.dims[0]    = inDims.elements();
    pVal.info.strides[0] = 1;
    for (int i = 1; i < 4; i++) {
        pVal.info.dims[i]    = 1;
        pVal.info.strides[i] = pVal.info.strides[i - 1] * pVal.info.dims[i - 1];
    }

    // Sort indices
    // sort_by_key<T, uint, isAscending>(*resVal, *resKey, val, key, 0);
    // kernel::sort0_by_key<T, uint, isAscending>(pVal, pKey);
    compute::command_queue c_queue(getQueue()());

    compute::buffer pKey_buf((*pKey.get())());
    compute::buffer pVal_buf((*pVal.data)());

    compute::buffer_iterator<type_t<T>> val0 =
        compute::make_buffer_iterator<type_t<T>>(pVal_buf, 0);
    compute::buffer_iterator<type_t<T>> valN =
        compute::make_buffer_iterator<type_t<T>>(pVal_buf, +pVal.info.dims[0]);
    compute::buffer_iterator<uint> key0 =
        compute::make_buffer_iterator<uint>(pKey_buf, 0);
    compute::buffer_iterator<uint> keyN =
        compute::make_buffer_iterator<uint>(pKey_buf, pKey.dims()[0]);
    if (isAscending) {
        compute::sort_by_key(val0, valN, key0, c_queue);
    } else {
        compute::sort_by_key(val0, valN, key0, compute::greater<type_t<T>>(),
                             c_queue);
    }

    // Needs to be ascending (true) in order to maintain the indices properly
    // kernel::sort0_by_key<uint, T, true>(pKey, pVal);
    compute::sort_by_key(key0, keyN, val0, c_queue);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void sort0(Param val, bool isAscending) {
    int higherDims = val.info.dims[1] * val.info.dims[2] * val.info.dims[3];
    // TODO Make a better heurisitic
    if (higherDims > 10)
        sortBatched<T>(val, 0, isAscending);
    else
        kernel::sort0Iterative<T>(val, isAscending);
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
