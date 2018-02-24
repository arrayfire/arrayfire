/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include <af/data.h>

#include <handle.hpp>
#include <topk.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>

using namespace detail;

namespace {

template<typename T>
af_err topk(af_array *v, af_array* i, const af_array in,
                    const int k, const int dim, const af_topk_function order)
{
    auto vals = createEmptyArray<T>(af::dim4());
    auto idxs = createEmptyArray<unsigned>(af::dim4());

    topk(vals, idxs, getArray<T>(in), k, dim, order);

    *v = getHandle<T>(vals);
    *i = getHandle<unsigned>(idxs);
    return AF_SUCCESS;
}
} //  namespace

af_err af_topk(af_array *values, af_array *indices, const af_array in,
                     const int k, const int dim, const af_topk_function order)
{
    try {
        af::topkFunction ord = (order == AF_TOPK_DEFAULT ? AF_TOPK_MAX : order);

        ArrayInfo inInfo = getInfo(in);

        ARG_ASSERT(1, (inInfo.ndims()>0));

        if (inInfo.elements() == 1) {
            dim_t dims[1] = {1};
            af_err errValue = af_constant(indices, 0, 1, dims, u32);
            return errValue==AF_SUCCESS ? af_retain_array(values, in) : errValue;
        }

        int rdim = dim;
        auto &inDims = inInfo.dims();

        if (rdim==-1) {
            for (dim_t d = 0; d < 4; d++) {
                if (inDims[d] > 1) {
                    rdim =  d;
                    break;
                }
            }
        }

        if (rdim!=0)
            AF_ERROR("topk is supported along dimenion 0 only.", AF_ERR_NOT_SUPPORTED);

        af_dtype type  = inInfo.getType();

        switch(type) {
            // TODO(umar): FIX RETURN VALUES HERE
            case f32: topk<float >(values, indices, in, k, rdim, ord); break;
            case f64: topk<double>(values, indices, in, k, rdim, ord); break;
            case u32: topk<uint  >(values, indices, in, k, rdim, ord); break;
            case s32: topk<int   >(values, indices, in, k, rdim, ord); break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
