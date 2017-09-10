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
#include <err_opencl.hpp>
#include <math.hpp>
#include <kernel/nearest_neighbour.hpp>
#include <transpose.hpp>

using af::dim4;
using cl::Device;

namespace opencl
{

template<typename T, typename To, af_match_type dist_type>
void nearest_neighbour_(Array<uint>& idx, Array<To>& dist,
                        const Array<T>& query, const Array<T>& train,
                        const uint dist_dim, const uint n_dist)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 outDims(n_dist, qDims[sample_dim]);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

    Array<T> queryT = dist_dim == 0 ? transpose(query, false) : query;
    Array<T> trainT = dist_dim == 0 ? transpose(train, false) : train;

    kernel::nearest_neighbour<T, To, dist_type>(idx, dist, queryT, trainT, 1, n_dist);

}

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist,
                       const Array<T>& query, const Array<T>& train,
                       const uint dist_dim, const uint n_dist,
                       const af_match_type dist_type)
{
    switch(dist_type) {
        case AF_SAD: nearest_neighbour_<T, To, AF_SAD>(idx, dist, query, train, dist_dim, n_dist); break;
        case AF_SSD: nearest_neighbour_<T, To, AF_SSD>(idx, dist, query, train, dist_dim, n_dist); break;
        case AF_SHD: nearest_neighbour_<T, To, AF_SHD>(idx, dist, query, train, dist_dim, n_dist); break;
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
INSTANTIATE(short , int)
INSTANTIATE(ushort, uint)
INSTANTIATE(uchar , uint)

INSTANTIATE(uintl, uint)    // For Hamming

}
