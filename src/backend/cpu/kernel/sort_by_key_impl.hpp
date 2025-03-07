/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_cpu.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel/sort_helper.hpp>
#include <math.hpp>
#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval, bool isAscending) {
    // Get pointers and initialize original index locations
    Tk *okey_ptr = okey.get();
    Tv *oval_ptr = oval.get();

    typedef IndexPair<Tk, Tv> CurrentPair;

    dim_t size = okey.dims(0);
    std::vector<CurrentPair> pairKeyVal(size);

    for (dim_t w = 0; w < okey.dims(3); w++) {
        dim_t okeyW = w * okey.strides(3);
        dim_t ovalW = w * oval.strides(3);

        for (dim_t z = 0; z < okey.dims(2); z++) {
            dim_t okeyWZ = okeyW + z * okey.strides(2);
            dim_t ovalWZ = ovalW + z * oval.strides(2);

            for (dim_t y = 0; y < okey.dims(1); y++) {
                dim_t okeyOffset = okeyWZ + y * okey.strides(1);
                dim_t ovalOffset = ovalWZ + y * oval.strides(1);

                Tk *okey_col_ptr = okey_ptr + okeyOffset;
                Tv *oval_col_ptr = oval_ptr + ovalOffset;

                for (dim_t x = 0; x < size; x++) {
                    pairKeyVal[x] =
                        std::make_tuple(okey_col_ptr[x], oval_col_ptr[x]);
                }

                if (isAscending) {
                    std::stable_sort(pairKeyVal.begin(), pairKeyVal.end(),
                                     IPCompare<Tk, Tv, true>());
                } else {
                    std::stable_sort(pairKeyVal.begin(), pairKeyVal.end(),
                                     IPCompare<Tk, Tv, false>());
                }

                for (unsigned x = 0; x < size; x++) {
                    okey_ptr[okeyOffset + x] = std::get<0>(pairKeyVal[x]);
                    oval_ptr[ovalOffset + x] = std::get<1>(pairKeyVal[x]);
                }
            }
        }
    }

    return;
}

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> okey, Param<Tv> oval, const int dim,
                      bool isAscending) {
    af::dim4 inDims = okey.dims();

    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    std::vector<uint> key(inDims.elements());
    // IOTA
    {
        af::dim4 dims = inDims;
        uint *out     = key.data();
        af::dim4 strides(1);
        for (int i = 1; i < 4; i++) strides[i] = strides[i - 1] * dims[i - 1];

        for (dim_t w = 0; w < dims[3]; w++) {
            dim_t offW = w * strides[3];
            dim_t okeyW =
                (w % seqDims[3]) * seqDims[0] * seqDims[1] * seqDims[2];
            for (dim_t z = 0; z < dims[2]; z++) {
                dim_t offWZ = offW + z * strides[2];
                dim_t okeyZ =
                    okeyW + (z % seqDims[2]) * seqDims[0] * seqDims[1];
                for (dim_t y = 0; y < dims[1]; y++) {
                    dim_t offWZY = offWZ + y * strides[1];
                    dim_t okeyY  = okeyZ + (y % seqDims[1]) * seqDims[0];
                    for (dim_t x = 0; x < dims[0]; x++) {
                        dim_t id = offWZY + x;
                        out[id]  = okeyY + (x % seqDims[0]);
                    }
                }
            }
        }
    }

    // initialize original index locations
    Tk *okey_ptr = okey.get();
    Tv *oval_ptr = oval.get();

    typedef KeyIndexPair<Tk, Tv> CurrentTuple;
    size_t size = okey.dims().elements();
    std::vector<CurrentTuple> tupleKeyValIdx(size);

    for (unsigned i = 0; i < size; i++) {
        tupleKeyValIdx[i] = std::make_tuple(okey_ptr[i], oval_ptr[i], key[i]);
    }

    if (isAscending) {
        std::stable_sort(tupleKeyValIdx.begin(), tupleKeyValIdx.end(),
                         KIPCompareV<Tk, Tv, true>());
    } else {
        std::stable_sort(tupleKeyValIdx.begin(), tupleKeyValIdx.end(),
                         KIPCompareV<Tk, Tv, false>());
    }

    std::stable_sort(tupleKeyValIdx.begin(), tupleKeyValIdx.end(),
                     KIPCompareK<Tk, Tv, true>());

    for (unsigned x = 0; x < okey.dims().elements(); x++) {
        okey_ptr[x] = std::get<0>(tupleKeyValIdx[x]);
        oval_ptr[x] = std::get<1>(tupleKeyValIdx[x]);
    }
}

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> okey, Param<Tv> oval, bool isAscending) {
    int higherDims = okey.dims(1) * okey.dims(2) * okey.dims(3);
    // TODO Make a better heurisitic
    if (higherDims > 4)
        kernel::sortByKeyBatched<Tk, Tv>(okey, oval, 0, isAscending);
    else
        kernel::sort0ByKeyIterative<Tk, Tv>(okey, oval, isAscending);
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
}  // namespace cpu
}  // namespace arrayfire
