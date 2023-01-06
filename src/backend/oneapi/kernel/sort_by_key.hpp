/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

// oneDPL headers should be included before standard headers
#define ONEDPL_USE_PREDEFINED_POLICIES 0
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <debug_oneapi.hpp>
#include <iota.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>

namespace oneapi {
namespace kernel {

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> pKey, Param<Tv> pVal, bool isAscending) {
    auto dpl_policy = oneapi::dpl::execution::make_device_policy(getQueue());

    for (int w = 0; w < pKey.info.dims[3]; w++) {
        int pKeyW = w * pKey.info.strides[3];
        int pValW = w * pVal.info.strides[3];
        for (int z = 0; z < pKey.info.dims[2]; z++) {
            int pKeyWZ = pKeyW + z * pKey.info.strides[2];
            int pValWZ = pValW + z * pVal.info.strides[2];
            for (int y = 0; y < pKey.info.dims[1]; y++) {
                int pKeyOffset = pKeyWZ + y * pKey.info.strides[1];
                int pValOffset = pValWZ + y * pVal.info.strides[1];

                auto key_begin    = oneapi::dpl::begin(*pKey.data) + pKeyOffset;
                auto key_end      = key_begin + pKey.info.dims[0];
                auto val_begin    = oneapi::dpl::begin(*pVal.data) + pValOffset;
                auto val_end      = val_begin + pVal.info.dims[0];

                auto zipped_begin = dpl::make_zip_iterator(key_begin, val_begin);
                auto zipped_end   = dpl::make_zip_iterator(key_end, val_end);

                // sort by key
                if (isAscending) {
                    std::sort(dpl_policy, zipped_begin, zipped_end, [](auto lhs, auto rhs) {
                        return std::get<0>(lhs) < std::get<0>(rhs);
                    });
                } else {
                    std::sort(dpl_policy, zipped_begin, zipped_end, [](auto lhs, auto rhs) {
                        return std::get<0>(lhs) > std::get<0>(rhs);
                    });
                }
            }
        }
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim, bool isAscending) {
    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pKey.dims[i];

    const dim_t elements = inDims.elements();

    // Sort dimension
    // tileDims * seqDims = inDims
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    Array<uint> Seq = iota<uint>(seqDims, tileDims);

    auto dpl_policy = oneapi::dpl::execution::make_device_policy(getQueue());

    auto seq_begin = oneapi::dpl::begin(*Seq.get());
    auto seq_end   = oneapi::dpl::end(*Seq.get());
    auto key_begin = oneapi::dpl::begin(*pKey.get());
    auto key_end   = oneapi::dpl::end(*pKey.get());
    auto val_begin = oneapi::dpl::begin(*pVal.data);
    auto val_end   = oneapi::dpl::end(*pVal.data);

    auto cKey = memAlloc<Tk>(elements);
    /*TODO: copy seq to cKey?
    const sycl::buffer<T> *A_buf = A.get();
    sycl::buffer<T> *out_buf     = out.get();

    getQueue()
        .submit([=](sycl::handler &h) {
            sycl::range rr(A.elements());
            sycl::id offset_id(offset);
            auto offset_acc_A =
                const_cast<sycl::buffer<T> *>(A_buf)->get_access(h, rr,
                                                                    offset_id);
            auto acc_out = out_buf->get_access(h);

            h.copy(offset_acc_A, acc_out);
        })
        .wait();
    */

}

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> pKey, Param<Tv> pVal, bool isAscending) {
    int higherDims = pKey.info.dims[1] * pKey.info.dims[2] * pKey.info.dims[3];
    // Batced sort performs 4x sort by keys
    // But this is only useful before GPU is saturated
    // The GPU is saturated at around 1000,000 integers
    // Call batched sort only if both conditions are met
    if (higherDims > 4 && pKey.info.dims[0] < 1000000) {
        kernel::sortByKeyBatched<Tk, Tv>(pKey, pVal, 0, isAscending);
    } else {
        kernel::sort0ByKeyIterative<Tk, Tv>(pKey, pVal, isAscending);
    }
}

} // namespace kernel
} // namespace oneapi