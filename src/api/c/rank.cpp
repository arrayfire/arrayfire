/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <qr.hpp>
#include <reduce.hpp>
#include <af/array.h>
#include <af/defines.h>
#include <af/lapack.h>

using af::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::logicOp;
using detail::reduce;
using detail::reduce_all;
using detail::scalar;
using detail::uint;

template<typename T>
static inline uint rank(const af_array in, double tol) {
    using BT          = typename af::dtype_traits<T>::base_type;
    const Array<T> In = getArray<T>(in);

    Array<BT> R = createEmptyArray<BT>(dim4());

    // Scoping to get rid of q, r and t as they are not necessary
    {
        Array<T> q = createEmptyArray<T>(dim4());
        Array<T> r = createEmptyArray<T>(dim4());
        Array<T> t = createEmptyArray<T>(dim4());
        qr(q, r, t, In);
        using detail::abs;

        R = abs<BT, T>(r);
    }

    Array<BT> val  = createValueArray<BT>(R.dims(), scalar<BT>(tol));
    Array<char> gt = logicOp<BT, af_gt_t>(R, val, val.dims());
    Array<char> at = reduce<af_or_t, char, char>(gt, 1);
    return reduce_all<af_notzero_t, char, uint>(at);
}

af_err af_rank(uint* out, const af_array in, const double tol) {
    try {
        const ArrayInfo& i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(0, out != nullptr);

        uint output = 0;
        if (i_info.ndims() != 0) {
            switch (type) {
                case f32: output = rank<float>(in, tol); break;
                case f64: output = rank<double>(in, tol); break;
                case c32: output = rank<cfloat>(in, tol); break;
                case c64: output = rank<cdouble>(in, tol); break;
                default: TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
