/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <debug_opencl.hpp>
#include <iota.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel/sort_helper.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>

AF_DEPRECATED_WARNINGS_OFF
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/functional/field.hpp>
#include <boost/compute/functional/get.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/types/pair.hpp>
AF_DEPRECATED_WARNINGS_ON

namespace compute = boost::compute;

using arrayfire::common::half;

template<typename Tk, typename Tv, bool isAscending>
inline boost::compute::function<bool(const std::pair<Tk, Tv>,
                                     const std::pair<Tk, Tv>)>
makeCompareFunction() {
    // Cannot use isAscending in BOOST_COMPUTE_FUNCTION
    if (isAscending) {
        BOOST_COMPUTE_FUNCTION(bool, IPCompare,
                               (std::pair<Tk, Tv> lhs, std::pair<Tk, Tv> rhs),
                               { return lhs.first < rhs.first; });
        return IPCompare;
    } else {
        BOOST_COMPUTE_FUNCTION(bool, IPCompare,
                               (std::pair<Tk, Tv> lhs, std::pair<Tk, Tv> rhs),
                               { return lhs.first > rhs.first; });
        return IPCompare;
    }
}

template<typename Tk>
inline boost::compute::function<Tk(Tk)> flipFunction() {
    BOOST_COMPUTE_FUNCTION(Tk, negateFn, (const Tk x), { return -x; });

    return negateFn;
}

#define INSTANTIATE_FLIP(TY, XMAX)                               \
    template<>                                                   \
    inline boost::compute::function<TY(TY)> flipFunction<TY>() { \
        BOOST_COMPUTE_FUNCTION(TY, negateFn, (const TY x),       \
                               { return XMAX - x; });            \
                                                                 \
        return negateFn;                                         \
    }

INSTANTIATE_FLIP(unsigned, UINT_MAX)
INSTANTIATE_FLIP(unsigned short, USHRT_MAX)
INSTANTIATE_FLIP(unsigned char, UCHAR_MAX)
INSTANTIATE_FLIP(cl_ulong, ULONG_MAX)

#undef INSTANTIATE_FLIP

namespace arrayfire {
namespace opencl {
namespace kernel {
static const int copyPairIter = 4;

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param pKey, Param pVal, bool isAscending) {
    compute::command_queue c_queue(getQueue()());

    compute::buffer pKey_buf((*pKey.data)());
    compute::buffer pVal_buf((*pVal.data)());

    for (int w = 0; w < pKey.info.dims[3]; w++) {
        int pKeyW = w * pKey.info.strides[3];
        int pValW = w * pVal.info.strides[3];
        for (int z = 0; z < pKey.info.dims[2]; z++) {
            int pKeyWZ = pKeyW + z * pKey.info.strides[2];
            int pValWZ = pValW + z * pVal.info.strides[2];
            for (int y = 0; y < pKey.info.dims[1]; y++) {
                int pKeyOffset = pKeyWZ + y * pKey.info.strides[1];
                int pValOffset = pValWZ + y * pVal.info.strides[1];

                compute::buffer_iterator<type_t<Tk>> start =
                    compute::make_buffer_iterator<type_t<Tk>>(pKey_buf,
                                                              pKeyOffset);
                compute::buffer_iterator<type_t<Tk>> end =
                    compute::make_buffer_iterator<type_t<Tk>>(
                        pKey_buf, pKeyOffset + pKey.info.dims[0]);
                compute::buffer_iterator<type_t<Tv>> vals =
                    compute::make_buffer_iterator<type_t<Tv>>(pVal_buf,
                                                              pValOffset);
                if (isAscending) {
                    compute::sort_by_key(start, end, vals, c_queue);
                } else {
                    compute::sort_by_key(start, end, vals,
                                         compute::greater<type_t<Tk>>(),
                                         c_queue);
                }
            }
        }
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk_, typename Tv_>
void sortByKeyBatched(Param pKey, Param pVal, const int dim, bool isAscending) {
    typedef type_t<Tk_> Tk;
    typedef type_t<Tv_> Tv;

    af::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pKey.info.dims[i];

    // Sort dimension
    // tileDims * seqDims = inDims
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    Array<unsigned> pSeq = iota<unsigned>(seqDims, tileDims);

    int elements = inDims.elements();

    // Flat - Not required since inplace and both are continuous
    // val.modDims(inDims.elements());
    // key.modDims(inDims.elements());

    // Sort indices
    // sort_by_key<T, Tv, isAscending>(*resVal, *resKey, val, key, 0);
    // kernel::sort0_by_key<T, Tv, isAscending>(pVal, pKey);
    compute::command_queue c_queue(getQueue()());
    compute::context c_context(getContext()());

    // Create buffer iterators for seq
    compute::buffer pSeq_buf((*pSeq.get())());
    compute::buffer_iterator<unsigned> seq0 =
        compute::make_buffer_iterator<unsigned>(pSeq_buf, 0);
    compute::buffer_iterator<unsigned> seqN =
        compute::make_buffer_iterator<unsigned>(pSeq_buf, elements);
    // Create buffer iterators for key and val
    compute::buffer pKey_buf((*pKey.data)());
    compute::buffer pVal_buf((*pVal.data)());
    compute::buffer_iterator<Tk> key0 =
        compute::make_buffer_iterator<Tk>(pKey_buf, 0);
    compute::buffer_iterator<Tk> keyN =
        compute::make_buffer_iterator<Tk>(pKey_buf, elements);
    compute::buffer_iterator<Tv> val0 =
        compute::make_buffer_iterator<Tv>(pVal_buf, 0);
    compute::buffer_iterator<Tv> valN =
        compute::make_buffer_iterator<Tv>(pVal_buf, elements);

    // Sort By Key for descending is stable in the reverse
    // (greater) order. Sorting in ascending with negated values
    // will give the right result
    if (!isAscending)
        compute::transform(key0, keyN, key0, flipFunction<Tk>(), c_queue);

    // Create a copy of the pKey buffer
    cl::Buffer* cKey = bufferAlloc(elements * sizeof(Tk));
    compute::buffer cKey_buf((*cKey)());
    compute::buffer_iterator<Tk> cKey0 =
        compute::make_buffer_iterator<Tk>(cKey_buf, 0);
    compute::buffer_iterator<Tk> cKeyN =
        compute::make_buffer_iterator<Tk>(cKey_buf, elements);
    compute::copy(key0, keyN, cKey0, c_queue);

    // FIRST SORT
    compute::sort_by_key(key0, keyN, seq0, c_queue);
    compute::sort_by_key(cKey0, cKeyN, val0, c_queue);

    // Create a copy of the seq buffer after first sort
    cl::Buffer* cSeq = bufferAlloc(elements * sizeof(unsigned));
    compute::buffer cSeq_buf((*cSeq)());
    compute::buffer_iterator<unsigned> cSeq0 =
        compute::make_buffer_iterator<unsigned>(cSeq_buf, 0);
    compute::buffer_iterator<unsigned> cSeqN =
        compute::make_buffer_iterator<unsigned>(cSeq_buf, elements);
    compute::copy(seq0, seqN, cSeq0, c_queue);

    // SECOND SORT
    // First call will sort key, second sort will sort val
    // Needs to be ascending (true) in order to maintain the indices properly
    // kernel::sort0_by_key<Tv, T, true>(pKey, pVal);
    compute::sort_by_key(seq0, seqN, key0, c_queue);
    compute::sort_by_key(cSeq0, cSeqN, val0, c_queue);

    // If descending, flip it back
    if (!isAscending)
        compute::transform(key0, keyN, key0, flipFunction<Tk>(), c_queue);

    CL_DEBUG_FINISH(getQueue());
    bufferFree(cSeq);
    bufferFree(cKey);
}

template<typename Tk, typename Tv>
void sort0ByKey(Param pKey, Param pVal, bool isAscending) {
    int higherDims = pKey.info.dims[1] * pKey.info.dims[2] * pKey.info.dims[3];
    // Batced sort performs 4x sort by keys
    // But this is only useful before GPU is saturated
    // The GPU is saturated at around 1000,000 integers
    // Call batched sort only if both conditions are met
    if (higherDims > 4 && pKey.info.dims[0] < 1000000)
        kernel::sortByKeyBatched<Tk, Tv>(pKey, pVal, 0, isAscending);
    else
        kernel::sort0ByKeyIterative<Tk, Tv>(pKey, pVal, isAscending);
}

#define INSTANTIATE(Tk, Tv)                                           \
    template void sort0ByKey<Tk, Tv>(Param okey, Param oval,          \
                                     bool isAscending);               \
    template void sort0ByKeyIterative<Tk, Tv>(Param okey, Param oval, \
                                              bool isAscending);      \
    template void sortByKeyBatched<Tk, Tv>(Param okey, Param oval,    \
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
    INSTANTIATE(Tk, uintl)   \
    INSTANTIATE(Tk, half)

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
