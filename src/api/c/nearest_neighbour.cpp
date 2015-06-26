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
#include <nearest_neighbour.hpp>

using af::dim4;
using namespace detail;

template<typename Ti, typename To>
static void nearest_neighbour(af_array* idx, af_array* dist,
        const af_array query, const af_array train,
        const dim_t dist_dim, const uint n_dist,
        const af_match_type dist_type)
{
    Array<uint> oIdxArray = createEmptyArray<uint>(af::dim4());
    Array<To>  oDistArray = createEmptyArray<To>(af::dim4());

    nearest_neighbour<Ti, To>(oIdxArray, oDistArray, getArray<Ti>(query), getArray<Ti>(train),
                              dist_dim, n_dist, dist_type);

    *idx  = getHandle<uint>(oIdxArray);
    *dist = getHandle<To>(oDistArray);
}

af_err af_nearest_neighbour(af_array* idx, af_array* dist,
        const af_array query, const af_array train,
        const dim_t dist_dim, const uint n_dist,
        const af_match_type dist_type)
{
    try {
        ArrayInfo qInfo = getInfo(query);
        ArrayInfo tInfo = getInfo(train);
        af_dtype qType  = qInfo.getType();
        af_dtype tType  = tInfo.getType();
        af::dim4 qDims  = qInfo.dims();
        af::dim4 tDims  = tInfo.dims();

        uint train_samples = (dist_dim == 0) ? 1 : 0;

        DIM_ASSERT(2, qDims[dist_dim] == tDims[dist_dim]);
        DIM_ASSERT(2, qDims[2] == 1 && qDims[3] == 1);
        DIM_ASSERT(3, tDims[2] == 1 && tDims[3] == 1);
        DIM_ASSERT(4, (dist_dim == 0 || dist_dim == 1));
        DIM_ASSERT(5, n_dist > 0 && n_dist <= (uint)tDims[train_samples]);
        ARG_ASSERT(6, dist_type == AF_SAD || dist_type == AF_SSD || dist_type == AF_SHD);
        TYPE_ASSERT(qType == tType);

        // For Hamming, only u8, u32 and u64 allowed.
        af_array oIdx;
        af_array oDist;

        if(dist_type == AF_SHD) {
            TYPE_ASSERT(qType == u8 || qType == u32 || qType == u64);
            switch(qType) {
                case u8:  nearest_neighbour<uchar, uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u32: nearest_neighbour<uint , uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u64: nearest_neighbour<uintl, uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                default : TYPE_ERROR(1, qType);
            }
        } else {
            switch(qType) {
                case f32: nearest_neighbour<float , float >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case f64: nearest_neighbour<double, double>(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case s32: nearest_neighbour<int   , int   >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u32: nearest_neighbour<uint  , uint  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case s64: nearest_neighbour<intl  , intl  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u64: nearest_neighbour<uintl , uintl >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u8:  nearest_neighbour<uchar , uint  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                default : TYPE_ERROR(1, qType);
            }
        }
        std::swap(*idx, oIdx);
        std::swap(*dist, oDist);
    }
    CATCHALL;

    return AF_SUCCESS;
}
