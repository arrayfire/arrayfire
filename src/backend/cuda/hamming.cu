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
#include <err_cuda.hpp>
#include <handle.hpp>
#include <kernel/hamming.hpp>
#include <kernel/transpose.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
void hamming_matcher(Array<uint>& idx, Array<uint>& dist,
                     const Array<T>& query, const Array<T>& train,
                     const uint dist_dim, const uint n_dist)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    const dim4 outDims(n_dist, qDims[sample_dim]);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<uint>(outDims);

    Array<T> queryT = query;
    Array<T> trainT = train;

    if (dist_dim == 0) {
        const dim4 queryTDims = dim4(qDims[1], qDims[0], qDims[2], qDims[3]);
        const dim4 trainTDims = dim4(tDims[1], tDims[0], tDims[2], tDims[3]);
        queryT = createEmptyArray<T>(queryTDims);
        trainT = createEmptyArray<T>(trainTDims);

        kernel::transpose<T, false>(queryT, query, query.ndims());
        kernel::transpose<T, false>(trainT, train, train.ndims());
    }

    kernel::hamming_matcher<T>(idx, dist, queryT, trainT, 1, n_dist);
}

#define INSTANTIATE(T)\
    template void hamming_matcher<T>(Array<uint>& idx, Array<uint>& dist,           \
                                     const Array<T>& query, const Array<T>& train,  \
                                     const uint dist_dim, const uint n_dist);

INSTANTIATE(uchar)
INSTANTIATE(uint)

}
