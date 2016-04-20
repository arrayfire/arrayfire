/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <math.hpp>
#include <algorithm>
#include <numeric>
#include <queue>
#include <err_cpu.hpp>
#include <functional>
#include <kernel/sort_helper.hpp>

namespace cpu
{
namespace kernel
{

template<typename Tk, typename Tv, bool isAscending>
void sort0ByKeyIterative(Array<Tk> okey, Array<Tv> oval)
{
    // Get pointers and initialize original index locations
    Tk *okey_ptr = okey.get();
    Tv *oval_ptr = oval.get();

    std::vector<IndexPair<Tk, Tv> > X;
    X.reserve(okey.dims()[0]);

    for(dim_t w = 0; w < okey.dims()[3]; w++) {
        dim_t okeyW = w * okey.strides()[3];
        dim_t ovalW = w * oval.strides()[3];

        for(dim_t z = 0; z < okey.dims()[2]; z++) {
            dim_t okeyWZ = okeyW + z * okey.strides()[2];
            dim_t ovalWZ = ovalW + z * oval.strides()[2];

            for(dim_t y = 0; y < okey.dims()[1]; y++) {

                dim_t okeyOffset = okeyWZ + y * okey.strides()[1];
                dim_t ovalOffset = ovalWZ + y * oval.strides()[1];

                X.clear();
                std::transform(okey_ptr + okeyOffset, okey_ptr + okeyOffset + okey.dims()[0],
                               oval_ptr + ovalOffset,
                               std::back_inserter(X),
                               [](Tk v_, Tv i_) { return std::make_pair(v_, i_); }
                               );

                std::stable_sort(X.begin(), X.end(), IPCompare<Tk, Tv, isAscending>());

                for(unsigned it = 0; it < X.size(); it++) {
                    okey_ptr[okeyOffset + it] = X[it].first;
                    oval_ptr[ovalOffset + it] = X[it].second;
                }
            }
        }
    }

    return;
}

template<typename Tk, typename Tv, bool isAscending, int dim>
void sortByKeyBatched(Array<Tk> okey, Array<Tv> oval)
{
    af::dim4 inDims = okey.dims();

    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim] = inDims[dim];
    seqDims[dim] = 1;

    uint* key = memAlloc<uint>(inDims.elements());
    // IOTA
    {
        af::dim4 dims    = inDims;
        uint* out        = key;
        af::dim4 strides(1);
        for(int i = 1; i < 4; i++)
            strides[i] = strides[i-1] * dims[i-1];

        for(dim_t w = 0; w < dims[3]; w++) {
            dim_t offW = w * strides[3];
            uint okeyW = (w % seqDims[3]) * seqDims[0] * seqDims[1] * seqDims[2];
            for(dim_t z = 0; z < dims[2]; z++) {
                dim_t offWZ = offW + z * strides[2];
                uint okeyZ = okeyW + (z % seqDims[2]) * seqDims[0] * seqDims[1];
                for(dim_t y = 0; y < dims[1]; y++) {
                    dim_t offWZY = offWZ + y * strides[1];
                    uint okeyY = okeyZ + (y % seqDims[1]) * seqDims[0];
                    for(dim_t x = 0; x < dims[0]; x++) {
                        dim_t id = offWZY + x;
                        out[id] = okeyY + (x % seqDims[0]);
                    }
                }
            }
        }
    }

    // initialize original index locations
    Tk *okey_ptr = okey.get();
    Tv *oval_ptr = oval.get();

    std::vector<KeyIndexPair<Tk, Tv> > X;
    X.reserve(okey.elements());

    for(unsigned i = 0; i < okey.elements(); i++) {
        X.push_back(std::make_pair(std::make_pair(okey_ptr[i], oval_ptr[i]), key[i]));
    }

    memFree(key); // key is no longer required

    std::stable_sort(X.begin(), X.end(), KIPCompareV<Tk, Tv, isAscending>());

    std::stable_sort(X.begin(), X.end(), KIPCompareK<Tk, Tv, true>());

    for(unsigned it = 0; it < okey.elements(); it++) {
        okey_ptr[it] = X[it].first.first;
        oval_ptr[it] = X[it].first.second;
    }

    return;
}

template<typename Tk, typename Tv, bool isAscending>
void sort0ByKey(Array<Tk> okey, Array<Tv> oval)
{
    int higherDims =  okey.dims()[1] * okey.dims()[2] * okey.dims()[3];
    // TODO Make a better heurisitic
    if(higherDims > 0)
        kernel::sortByKeyBatched<Tk, Tv, isAscending, 0>(okey, oval);
    else
        kernel::sort0ByKeyIterative<Tk, Tv, isAscending>(okey, oval);
}

}
}
