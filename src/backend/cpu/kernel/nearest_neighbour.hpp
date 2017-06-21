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

namespace cpu
{
namespace kernel
{

#if defined(_WIN32) || defined(_MSC_VER)

#include <intrin.h>
#define __builtin_popcount __popcnt

#endif

template<typename T, typename To, af_match_type dist_type>
struct dist_op
{
    To operator()(T v1, T v2)
    {
        return v1 - v2;     // Garbage distance
    }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SAD>
{
    To operator()(T v1, T v2)
    {
        return std::abs((double)v1 - (double)v2);
    }
};

template<typename T, typename To>
struct dist_op<T, To, AF_SSD>
{
    To operator()(T v1, T v2)
    {
        return (v1 - v2) * (v1 - v2);
    }
};

template<typename To>
struct dist_op<uint, To, AF_SHD>
{
    To operator()(uint v1, uint v2)
    {
        return __builtin_popcount(v1 ^ v2);
    }
};

template<typename To>
struct dist_op<uintl, To, AF_SHD>
{
    To operator()(uintl v1, uintl v2)
    {
        return __builtin_popcount(v1 ^ v2);
    }
};

template<typename To>
struct dist_op<uchar, To, AF_SHD>
{
    To operator()(uchar v1, uchar v2)
    {
        return __builtin_popcount(v1 ^ v2);
    }
};

template<typename To>
struct dist_op<ushort, To, AF_SHD>
{
    To operator()(ushort v1, ushort v2)
    {
        return __builtin_popcount(v1 ^ v2);
    }
};

template<typename T, typename To, af_match_type dist_type>
void nearest_neighbour(Param<uint> idx, Param<To> dist,
                       CParam<T> query, CParam<T> train,
                       const uint dist_dim, const uint n_dist)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims;
    const dim4 tDims = train.dims;

    const unsigned distLength = qDims[dist_dim];
    const unsigned nQuery = qDims[sample_dim];
    const unsigned nTrain = tDims[sample_dim];

    const T* qPtr = query.get();
    const T* tPtr = train.get();
    uint* iPtr = idx.get();
    To* dPtr = dist.get();

    dist_op<T, To, dist_type> op;

    for (unsigned i = 0; i < nQuery; i++) {
        To best_dist = maxval<To>();
        unsigned best_idx  = 0;

        for (unsigned j = 0; j < nTrain; j++) {
            To local_dist = 0;
            for (unsigned k = 0; k < distLength; k++) {
                size_t qIdx, tIdx;
                if (sample_dim == 0) {
                    qIdx = k * qDims[0] + i;
                    tIdx = k * tDims[0] + j;
                }
                else {
                    qIdx = i * qDims[0] + k;
                    tIdx = j * tDims[0] + k;
                }

                local_dist += op(qPtr[qIdx], tPtr[tIdx]);
            }

            if (local_dist < best_dist) {
                best_dist = local_dist;
                best_idx  = j;
            }
        }

        size_t oIdx;
        oIdx = i;
        iPtr[oIdx] = best_idx;
        dPtr[oIdx] = best_dist;
    }
}

}
}
