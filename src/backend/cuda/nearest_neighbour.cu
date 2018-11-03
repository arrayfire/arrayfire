/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <kernel/nearest_neighbour.hpp>
#include <topk.hpp>
#include <transpose.hpp>

using af::dim4;

namespace cuda
{

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist,
                     const Array<T>& query, const Array<T>& train,
                     const uint dist_dim, const uint n_dist,
                     const af_match_type dist_type)
{
    uint  sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    const dim4 outDims(n_dist, qDims[sample_dim]);
    const dim4 distDims(tDims[sample_dim], qDims[sample_dim]);

    Array<To> tmp_dists = createEmptyArray<To>(distDims);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

    Array<T> queryT = dist_dim == 0 ? transpose(query, false) : query;
    Array<T> trainT = dist_dim == 0 ? transpose(train, false) : train;

    switch(dist_type) {
        case AF_SAD: kernel::all_distances<T, To, AF_SAD>(tmp_dists, queryT, trainT, 1, n_dist);
                     break;
        case AF_SSD: kernel::all_distances<T, To, AF_SSD>(tmp_dists, queryT, trainT, 1, n_dist);
                     break;
        case AF_SHD: kernel::all_distances<T, To, AF_SHD>(tmp_dists, queryT, trainT, 1, n_dist);
                     break;
        default: AF_ERROR("Unsupported dist_type", AF_ERR_NOT_CONFIGURED);
    }

    topk(dist, idx, tmp_dists, n_dist, 0, AF_TOPK_MIN);
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
INSTANTIATE(short , int)
INSTANTIATE(ushort, uint)

INSTANTIATE(uintl, uint)    // For Hamming

}
