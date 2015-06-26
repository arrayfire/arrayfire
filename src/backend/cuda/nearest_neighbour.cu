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
#include <kernel/nearest_neighbour.hpp>
#include <kernel/transpose.hpp>

using af::dim4;

namespace cuda
{

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist,
                     const Array<T>& query, const Array<T>& train,
                     const uint dist_dim, const uint n_dist,
                     const af_match_type dist_type)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    const dim4 outDims(n_dist, qDims[sample_dim]);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

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

    switch(dist_type) {
        case AF_SAD: kernel::nearest_neighbour<T, To, AF_SAD>(idx, dist, queryT, trainT, 1, n_dist);
                     break;
        case AF_SSD: kernel::nearest_neighbour<T, To, AF_SSD>(idx, dist, queryT, trainT, 1, n_dist);
                     break;
        case AF_SHD: kernel::nearest_neighbour<T, To, AF_SHD>(idx, dist, queryT, trainT, 1, n_dist);
                     break;
        default: AF_ERROR("Unsupported dist_type", AF_ERR_NOT_CONFIGURED);
    }
}

#define INSTANTIATE(T, To)                                                              \
    template void nearest_neighbour<T, To>(Array<uint>& idx, Array<To>& dist,           \
                                         const Array<T>& query, const Array<T>& train,  \
                                         const uint dist_dim, const uint n_dist,        \
                                         const af_match_type dist_type);

INSTANTIATE(float , float)
INSTANTIATE(double, double)
INSTANTIATE(int   , int)
INSTANTIATE(uint  , uint)
INSTANTIATE(intl  , intl)
INSTANTIATE(uintl , uintl)
INSTANTIATE(uchar , uint)

INSTANTIATE(uintl, uint)    // For Hamming

}
