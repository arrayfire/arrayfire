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
#include <kernel/hamming.hpp>
#include <kernel/transpose.hpp>

using af::dim4;
using cl::Device;

namespace opencl
{

static const unsigned THREADS = 256;

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
        switch (feat_len) {
        case 1:
            kernel::hamming_matcher<T, true , 1 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 2:
            kernel::hamming_matcher<T, true , 2 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 4:
            kernel::hamming_matcher<T, true , 4 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 8:
            kernel::hamming_matcher<T, true , 8 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 16:
            kernel::hamming_matcher<T, true , 16>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 32:
            kernel::hamming_matcher<T, true , 32>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 64:
            kernel::hamming_matcher<T, true , 64>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        default:
            kernel::hamming_matcher<T, true , 0 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        }
    } else {
        switch (feat_len) {
        case 1:
            kernel::hamming_matcher<T, false, 1 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 2:
            kernel::hamming_matcher<T, false, 2 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 4:
            kernel::hamming_matcher<T, false, 4 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 8:
            kernel::hamming_matcher<T, false, 8 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 16:
            kernel::hamming_matcher<T, false, 16>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 32:
            kernel::hamming_matcher<T, false, 32>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        case 64:
            kernel::hamming_matcher<T, false, 64>(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        default:
            kernel::hamming_matcher<T, false, 0 >(idx, dist, queryT, trainT, 1, n_dist, lmem_sz);
            break;
        }
    }
}

#define INSTANTIATE(T)\
    template void hamming_matcher<T>(Array<uint>& idx, Array<uint>& dist,           \
                                     const Array<T>& query, const Array<T>& train,  \
                                     const uint dist_dim, const uint n_dist);

INSTANTIATE(uchar)
INSTANTIATE(uint)

}
