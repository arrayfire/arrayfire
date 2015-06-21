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
#include <af/vision.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <hamming.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static void hamming_matcher(af_array* idx, af_array* dist, const af_array query, const af_array train, const dim_t dist_dim, const uint n_dist)
{
    Array<uint> oIdxArray = createEmptyArray<uint>(af::dim4());
    Array<uint> oDistArray = createEmptyArray<uint>(af::dim4());

    hamming_matcher<T>(oIdxArray, oDistArray, getArray<T>(query), getArray<T>(train), dist_dim, n_dist);

    *idx  = getHandle<uint>(oIdxArray);
    *dist = getHandle<uint>(oDistArray);
}

af_err af_hamming_matcher(af_array* idx, af_array* dist, const af_array query, const af_array train, const dim_t dist_dim, const uint n_dist)
{
    try {
        ArrayInfo qInfo = getInfo(query);
        ArrayInfo tInfo = getInfo(train);
        af_dtype qType  = qInfo.getType();
        af_dtype tType  = tInfo.getType();
        af::dim4 qDims  = qInfo.dims();
        af::dim4 tDims  = tInfo.dims();

        uint train_samples = (dist_dim == 0) ? 1 : 0;

        DIM_ASSERT(3, qDims[dist_dim] == tDims[dist_dim]);
        DIM_ASSERT(3, qDims[2] == 1 && qDims[3] == 1);
        DIM_ASSERT(3, qType == tType);
        DIM_ASSERT(4, tDims[2] == 1 && tDims[3] == 1);
        DIM_ASSERT(5, (dist_dim == 0 || dist_dim == 1));
        DIM_ASSERT(6, n_dist > 0 && n_dist <= (uint)tDims[train_samples]);

        af_array oIdx;
        af_array oDist;
        switch(qType) {
            case u8:  hamming_matcher<uchar>(&oIdx, &oDist, query, train, dist_dim, n_dist); break;
            case u32: hamming_matcher<uint >(&oIdx, &oDist, query, train, dist_dim, n_dist); break;
            default : TYPE_ERROR(1, qType);
        }
        std::swap(*idx, oIdx);
        std::swap(*dist, oDist);
    }
    CATCHALL;

    return AF_SUCCESS;
}
