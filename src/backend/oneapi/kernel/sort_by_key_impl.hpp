/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
// temporary ignores for DPL internals
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

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
#include <types.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

using arrayfire::common::half;

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> pKey, Param<Tv> pVal, bool isAscending) {
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    for (int w = 0; w < pKey.info.dims[3]; w++) {
        int pKeyW = w * pKey.info.strides[3];
        int pValW = w * pVal.info.strides[3];
        for (int z = 0; z < pKey.info.dims[2]; z++) {
            int pKeyWZ = pKeyW + z * pKey.info.strides[2];
            int pValWZ = pValW + z * pVal.info.strides[2];
            for (int y = 0; y < pKey.info.dims[1]; y++) {
                int pKeyOffset = pKeyWZ + y * pKey.info.strides[1];
                int pValOffset = pValWZ + y * pVal.info.strides[1];

                auto key_begin =
                    ::oneapi::dpl::begin(
                        pKey.data->template reinterpret<compute_t<Tk>>()) +
                    pKeyOffset;
                auto key_end   = key_begin + pKey.info.dims[0];
                auto val_begin = ::oneapi::dpl::begin(*pVal.data) + pValOffset;
                auto val_end   = val_begin + pVal.info.dims[0];

                auto zipped_begin =
                    ::oneapi::dpl::make_zip_iterator(key_begin, val_begin);
                auto zipped_end =
                    ::oneapi::dpl::make_zip_iterator(key_end, val_end);

                // sort by key
                if (isAscending) {
                    std::sort(dpl_policy, zipped_begin, zipped_end,
                              [](auto lhs, auto rhs) {
                                  return std::get<0>(lhs) < std::get<0>(rhs);
                              });
                } else {
                    std::sort(dpl_policy, zipped_begin, zipped_end,
                              [](auto lhs, auto rhs) {
                                  return std::get<0>(lhs) > std::get<0>(rhs);
                              });
                }
            }
        }
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim,
                      bool isAscending) {
    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pKey.info.dims[i];

    const dim_t elements = inDims.elements();

    // Sort dimension
    // tileDims * seqDims = inDims
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    Array<uint> Seq = iota<uint>(seqDims, tileDims);

    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    // set up iterators for seq, key, val, and new cKey
    auto seq_begin = ::oneapi::dpl::begin(*Seq.get());
    auto seq_end   = seq_begin + elements;
    auto key_begin =
        ::oneapi::dpl::begin(pKey.data->template reinterpret<compute_t<Tk>>());
    auto key_end = key_begin + elements;

    auto val_begin = ::oneapi::dpl::begin(*pVal.data);
    auto val_end   = val_begin + elements;

    auto cKey = memAlloc<Tk>(elements);
    auto cKey_get = cKey.get();
    getQueue().submit([&](sycl::handler &h) {
        h.copy(pKey.data->template reinterpret<compute_t<Tk>>().get_access(
                   h, elements),
               cKey_get->template reinterpret<compute_t<Tk>>().get_access(
                   h, elements));
    });
    auto ckey_begin =
        ::oneapi::dpl::begin(cKey.get()->template reinterpret<compute_t<Tk>>());
    auto ckey_end = ckey_begin + elements;

    {
        auto zipped_begin_KV  = dpl::make_zip_iterator(key_begin, val_begin);
        auto zipped_end_KV    = dpl::make_zip_iterator(key_end, val_end);
        auto zipped_begin_cKS = dpl::make_zip_iterator(ckey_begin, seq_begin);
        auto zipped_end_cKS   = dpl::make_zip_iterator(ckey_end, seq_end);
        if (isAscending) {
            std::sort(dpl_policy, zipped_begin_KV, zipped_end_KV,
                      [](auto lhs, auto rhs) {
                          return std::get<0>(lhs) < std::get<0>(rhs);
                      });
            std::sort(dpl_policy, zipped_begin_cKS, zipped_end_cKS,
                      [](auto lhs, auto rhs) {
                          return std::get<0>(lhs) < std::get<0>(rhs);
                      });
        } else {
            std::sort(dpl_policy, zipped_begin_KV, zipped_end_KV,
                      [](auto lhs, auto rhs) {
                          return std::get<0>(lhs) > std::get<0>(rhs);
                      });
            std::sort(dpl_policy, zipped_begin_cKS, zipped_end_cKS,
                      [](auto lhs, auto rhs) {
                          return std::get<0>(lhs) > std::get<0>(rhs);
                      });
        }
    }

    auto Seq_get = Seq.get();
    auto cSeq = memAlloc<uint>(elements);
    auto cSeq_get = cSeq.get();
    getQueue().submit([&](sycl::handler &h) {
        h.copy(Seq_get->get_access(h, elements),
               cSeq_get->get_access(h, elements));
    });
    auto cseq_begin = ::oneapi::dpl::begin(*cSeq.get());
    auto cseq_end   = cseq_begin + elements;

    {
        auto zipped_begin_SV  = dpl::make_zip_iterator(seq_begin, val_begin);
        auto zipped_end_SV    = dpl::make_zip_iterator(seq_end, val_end);
        auto zipped_begin_cSK = dpl::make_zip_iterator(cseq_begin, key_begin);
        auto zipped_end_cSK   = dpl::make_zip_iterator(cseq_end, key_end);
        std::sort(dpl_policy, zipped_begin_SV, zipped_end_SV,
                  [](auto lhs, auto rhs) {
                      return std::get<0>(lhs) < std::get<0>(rhs);
                  });
        std::sort(dpl_policy, zipped_begin_cSK, zipped_end_cSK,
                  [](auto lhs, auto rhs) {
                      return std::get<0>(lhs) < std::get<0>(rhs);
                  });
    }
}

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> pKey, Param<Tv> pVal, bool isAscending) {
    int higherDims = pKey.info.dims[1] * pKey.info.dims[2] * pKey.info.dims[3];
    // Batched sort performs 4x sort by keys
    // But this is only useful before GPU is saturated
    // The GPU is saturated at around 1000,000 integers
    // Call batched sort only if both conditions are met
    if (higherDims > 4 && pKey.info.dims[0] < 1000000) {
        kernel::sortByKeyBatched<Tk, Tv>(pKey, pVal, 0, isAscending);
    } else {
        kernel::sort0ByKeyIterative<Tk, Tv>(pKey, pVal, isAscending);
    }
}

#define INSTANTIATE(Tk, Tv)                                                   \
    template void sort0ByKey<Tk, Tv>(Param<Tk> okey, Param<Tv> oval,          \
                                     bool isAscending);                       \
    template void sort0ByKeyIterative<Tk, Tv>(Param<Tk> okey, Param<Tv> oval, \
                                              bool isAscending);              \
    template void sortByKeyBatched<Tk, Tv>(Param<Tk> okey, Param<Tv> oval,    \
                                           const int dim, bool isAscending);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic pop
#endif
