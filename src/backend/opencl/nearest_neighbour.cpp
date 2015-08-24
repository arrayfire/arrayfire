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
#include <err_opencl.hpp>
#include <handle.hpp>
#include <kernel/nearest_neighbour.hpp>
#include <kernel/transpose.hpp>

using af::dim4;
using cl::Device;

namespace opencl
{

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type>
void nearest_neighbour_(Array<uint>& idx, Array<To>& dist,
                     const Array<T>& query, const Array<T>& train,
                     const uint dist_dim, const uint n_dist)
{
    uint sample_dim = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    const dim4 outDims(n_dist, qDims[sample_dim]);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

    const unsigned feat_len = qDims[dist_dim];

    // Determine maximum feat_len capable of using shared memory (faster)
    cl_ulong avail_lmem = getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    size_t lmem_predef = 2 * THREADS * sizeof(unsigned) + feat_len * sizeof(T);
    size_t ltrain_sz = THREADS * feat_len * sizeof(T);
    bool use_lmem = (avail_lmem >= (lmem_predef + ltrain_sz)) ? true : false;
    size_t lmem_sz = (use_lmem) ? lmem_predef + ltrain_sz : lmem_predef;

    Array<T> queryT = query;
    Array<T> trainT = train;

    if (dist_dim == 0) {
        const dim4 queryTDims = dim4(qDims[1], qDims[0], qDims[2], qDims[3]);
        const dim4 trainTDims = dim4(tDims[1], tDims[0], tDims[2], tDims[3]);
        queryT = createEmptyArray<T>(queryTDims);
        trainT = createEmptyArray<T>(trainTDims);

        bool queryIs32Multiple = false;
        if (qDims[0] % 32 == 0 && qDims[1] % 32 == 0)
            queryIs32Multiple = true;

        bool trainIs32Multiple = false;
        if (tDims[0] % 32 == 0 && tDims[1] % 32 == 0)
        trainIs32Multiple = true;

        if (queryIs32Multiple)
            kernel::transpose<T, false, true >(queryT, query);
        else
            kernel::transpose<T, false, false>(queryT, query);

        if (trainIs32Multiple)
            kernel::transpose<T, false, true >(trainT, train);
        else
            kernel::transpose<T, false, false>(trainT, train);
    }

    if (use_lmem) {
        kernel::nearest_neighbour<T, To, dist_type, true >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
    } else {
        kernel::nearest_neighbour<T, To, dist_type, false>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
    }
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
INSTANTIATE(uchar , uint)

INSTANTIATE(uintl, uint)    // For Hamming

}
