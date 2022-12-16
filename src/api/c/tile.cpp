/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/tile.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <unary.hpp>
#include <af/arith.h>
#include <af/data.h>

using af::dim4;
using arrayfire::common::half;
using arrayfire::common::tile;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::unaryOp;
using detail::ushort;

template<typename T>
static inline af_array tile(const af_array in, const af::dim4 &tileDims) {
    return getHandle(arrayfire::common::tile<T>(getArray<T>(in), tileDims));
}

af_err af_tile(af_array *out, const af_array in, const af::dim4 &tileDims) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (info.ndims() == 0) { return af_retain_array(out, in); }
        DIM_ASSERT(1, info.dims().elements() > 0);
        DIM_ASSERT(2, tileDims.elements() > 0);

        af_array output;

        switch (type) {
            case f32: output = tile<float>(in, tileDims); break;
            case c32: output = tile<cfloat>(in, tileDims); break;
            case f64: output = tile<double>(in, tileDims); break;
            case c64: output = tile<cdouble>(in, tileDims); break;
            case b8: output = tile<char>(in, tileDims); break;
            case s32: output = tile<int>(in, tileDims); break;
            case u32: output = tile<uint>(in, tileDims); break;
            case s64: output = tile<intl>(in, tileDims); break;
            case u64: output = tile<uintl>(in, tileDims); break;
            case s16: output = tile<short>(in, tileDims); break;
            case u16: output = tile<ushort>(in, tileDims); break;
            case u8: output = tile<uchar>(in, tileDims); break;
            case f16: output = tile<half>(in, tileDims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_tile(af_array *out, const af_array in, const unsigned x,
               const unsigned y, const unsigned z, const unsigned w) {
    af::dim4 tileDims(x, y, z, w);
    return af_tile(out, in, tileDims);
}
