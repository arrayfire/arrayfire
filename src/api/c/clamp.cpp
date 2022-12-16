/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <logic.hpp>
#include <optypes.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>

using af::dim4;
using arrayfire::common::half;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline af_array clampOp(const af_array in, const af_array lo,
                               const af_array hi, const dim4& odims) {
    const Array<T> L = castArray<T>(lo);
    const Array<T> H = castArray<T>(hi);
    const Array<T> I = castArray<T>(in);
    return getHandle(
        arithOp<T, af_min_t>(arithOp<T, af_max_t>(I, L, odims), H, odims));
}

af_err af_clamp(af_array* out, const af_array in, const af_array lo,
                const af_array hi, const bool batch) {
    try {
        const ArrayInfo& linfo = getInfo(lo);
        const ArrayInfo& hinfo = getInfo(hi);
        const ArrayInfo& iinfo = getInfo(in);

        DIM_ASSERT(2, linfo.dims() == hinfo.dims());
        TYPE_ASSERT(linfo.getType() == hinfo.getType());

        dim4 odims           = getOutDims(iinfo.dims(), linfo.dims(), batch);
        const af_dtype otype = implicit(iinfo.getType(), linfo.getType());

        af_array res;
        switch (otype) {
            case f32: res = clampOp<float>(in, lo, hi, odims); break;
            case f64: res = clampOp<double>(in, lo, hi, odims); break;
            case c32: res = clampOp<cfloat>(in, lo, hi, odims); break;
            case c64: res = clampOp<cdouble>(in, lo, hi, odims); break;
            case s32: res = clampOp<int>(in, lo, hi, odims); break;
            case u32: res = clampOp<uint>(in, lo, hi, odims); break;
            case u8: res = clampOp<uchar>(in, lo, hi, odims); break;
            case b8: res = clampOp<char>(in, lo, hi, odims); break;
            case s64: res = clampOp<intl>(in, lo, hi, odims); break;
            case u64: res = clampOp<uintl>(in, lo, hi, odims); break;
            case s16: res = clampOp<short>(in, lo, hi, odims); break;
            case u16: res = clampOp<ushort>(in, lo, hi, odims); break;
            case f16: res = clampOp<half>(in, lo, hi, odims); break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}
