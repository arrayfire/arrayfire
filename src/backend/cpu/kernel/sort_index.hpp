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
#include <err_cpu.hpp>
#include <functional>
#include <kernel/sort_helper.hpp>

namespace cpu
{
namespace kernel
{

template<typename T, bool isAscending>
void sort0IndexIterative(Array<T> val, Array<uint> idx)
{
    // initialize original index locations
    uint *idx_ptr = idx.get();
       T *val_ptr = val.get();

    std::vector<IndexPair<T> > X;
    X.reserve(val.dims()[0]);

    for(dim_t w = 0; w < val.dims()[3]; w++) {
        dim_t valW = w * val.strides()[3];
        dim_t idxW = w * idx.strides()[3];
        for(dim_t z = 0; z < val.dims()[2]; z++) {
            dim_t valWZ = valW + z * val.strides()[2];
            dim_t idxWZ = idxW + z * idx.strides()[2];
            for(dim_t y = 0; y < val.dims()[1]; y++) {
                dim_t valOffset = valWZ + y * val.strides()[1];
                dim_t idxOffset = idxWZ + y * idx.strides()[1];

                X.clear();
                std::transform(val_ptr + valOffset, val_ptr + valOffset + val.dims()[0],
                               idx_ptr + idxOffset,
                               std::back_inserter(X),
                               [](T v_, uint i_) { return std::make_pair(v_, i_); }
                               );

                //comp_ptr = &X.front();
                std::stable_sort(X.begin(), X.end(), IPCompare<T, isAscending>());

                for(unsigned it = 0; it < X.size(); it++) {
                    val_ptr[valOffset + it] = X[it].first;
                    idx_ptr[idxOffset + it] = X[it].second;
                }
            }
        }
    }

    return;
}

template<typename T, bool isAscending, int dim>
void sortIndexBatched(Array<T> val, Array<uint> idx)
{
    af::dim4 inDims = val.dims();

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
            T valW = (w % seqDims[3]) * seqDims[0] * seqDims[1] * seqDims[2];
            for(dim_t z = 0; z < dims[2]; z++) {
                dim_t offWZ = offW + z * strides[2];
                T valZ = valW + (z % seqDims[2]) * seqDims[0] * seqDims[1];
                for(dim_t y = 0; y < dims[1]; y++) {
                    dim_t offWZY = offWZ + y * strides[1];
                    T valY = valZ + (y % seqDims[1]) * seqDims[0];
                    for(dim_t x = 0; x < dims[0]; x++) {
                        dim_t id = offWZY + x;
                        out[id] = valY + (x % seqDims[0]);
                    }
                }
            }
        }
    }

    // initialize original index locations
    uint *idx_ptr = idx.get();
       T *val_ptr = val.get();

    std::vector<KeyIndexPair<T> > X;
    X.reserve(val.elements());

    for(unsigned i = 0; i < val.elements(); i++) {
        X.push_back(std::make_pair(std::make_pair(val_ptr[i], idx_ptr[i]), key[i]));
    }

    memFree(key); // key is no longer required

    std::stable_sort(X.begin(), X.end(), KIPCompareV<T, isAscending>());

    std::stable_sort(X.begin(), X.end(), KIPCompareK<T, true>());

    for(unsigned it = 0; it < val.elements(); it++) {
        val_ptr[it] = X[it].first.first;
        idx_ptr[it] = X[it].first.second;
    }

    return;
}

template<typename T, bool isAscending>
void sort0Index(Array<T> val, Array<unsigned> idx)
{
    int higherDims =  val.dims()[1] * val.dims()[2] * val.dims()[3];
    // TODO Make a better heurisitic
    if(higherDims > 0)
        kernel::sortIndexBatched<T, isAscending, 0>(val, idx);
    else
        kernel::sort0IndexIterative<T, isAscending>(val, idx);
}

}
}
