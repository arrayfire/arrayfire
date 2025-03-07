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
#include <traits.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
void sort0Iterative(Param<T> val, bool isAscending) {
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());
    for (int w = 0; w < val.info.dims[3]; w++) {
        int valW = w * val.info.strides[3];
        for (int z = 0; z < val.info.dims[2]; z++) {
            int valWZ = valW + z * val.info.strides[2];
            for (int y = 0; y < val.info.dims[1]; y++) {
                int valOffset = valWZ + y * val.info.strides[1];

                auto buf_begin = ::oneapi::dpl::begin(*val.data) + valOffset;
                auto buf_end   = buf_begin + val.info.dims[0];
                if (isAscending) {
                    std::sort(dpl_policy, buf_begin, buf_end,
                              [](auto lhs, auto rhs) { return lhs < rhs; });
                    // std::less<T>()); // mangled name errors in icx for now
                } else {
                    std::sort(dpl_policy, buf_begin, buf_end,
                              [](auto lhs, auto rhs) { return lhs > rhs; });
                    // std::greater<T>()); // mangled name errors in icx for now
                }
            }
        }
    }
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
void sortBatched(Param<T> pVal, int dim, bool isAscending) {
    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pVal.info.dims[i];

    // Sort dimension
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
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
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    auto key_begin    = ::oneapi::dpl::begin(*pKey.get());
    auto key_end      = key_begin + pKey.dims()[0];
    auto val_begin    = ::oneapi::dpl::begin(*pVal.data);
    auto val_end      = val_begin + pVal.info.dims[0];
    auto zipped_begin = dpl::make_zip_iterator(key_begin, val_begin);
    auto zipped_end   = dpl::make_zip_iterator(key_end, val_end);

    // sort values first
    if (isAscending) {
        std::sort(dpl_policy, zipped_begin, zipped_end, [](auto lhs, auto rhs) {
            return std::get<1>(lhs) < std::get<1>(rhs);
        });
    } else {
        std::sort(dpl_policy, zipped_begin, zipped_end, [](auto lhs, auto rhs) {
            return std::get<1>(lhs) > std::get<1>(rhs);
        });
    }
    // sort according to keys second
    std::sort(dpl_policy, zipped_begin, zipped_end, [](auto lhs, auto rhs) {
        return std::get<0>(lhs) < std::get<0>(rhs);
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
void sort0(Param<T> val, bool isAscending) {
    int higherDims = val.info.dims[1] * val.info.dims[2] * val.info.dims[3];
    // TODO Make a better heurisitic
    if (higherDims > 10)
        sortBatched<T>(val, 0, isAscending);
    else
        sort0Iterative<T>(val, isAscending);
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
