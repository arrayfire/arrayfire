/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <handle.hpp>

using af::dim4;

namespace cpu
{

#if defined(_WIN32) || defined(_MSC_VER)

#include <intrin.h>
#define __builtin_popcount __popcnt

#endif

template<typename T>
inline uint hamming_distance(T v1, T v2)
{
    return __builtin_popcount(v1 ^ v2);
}

template<typename T>
void hamming_matcher(Array<uint>& idx, Array<uint>& dist,
                     const Array<T>& query, const Array<T>& train,
                     const uint dist_dim, const uint n_dist)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    if (n_dist > 1) {
        CPU_NOT_SUPPORTED();
    }

    const unsigned distLength = qDims[dist_dim];
    const unsigned nQuery = qDims[sample_dim];
    const unsigned nTrain = tDims[sample_dim];

    const dim4 outDims(n_dist, nQuery);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<uint>(outDims);

    const T* qPtr = query.get();
    const T* tPtr = train.get();
    uint* iPtr = idx.get();
    uint* dPtr = dist.get();

    for (unsigned i = 0; i < nQuery; i++) {
        unsigned best_dist = limit_max<unsigned>();
        unsigned best_idx  = 0;

        for (unsigned j = 0; j < nTrain; j++) {
            unsigned local_dist = 0;
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

                local_dist += hamming_distance(qPtr[qIdx], tPtr[tIdx]);
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

#define INSTANTIATE(T)\
    template void hamming_matcher<T>(Array<uint>& idx, Array<uint>& dist,           \
                                     const Array<T>& query, const Array<T>& train,  \
                                     const uint dist_dim, const uint n_dist);

INSTANTIATE(uchar)
INSTANTIATE(uint)

}
