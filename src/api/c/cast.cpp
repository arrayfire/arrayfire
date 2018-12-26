/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/ArrayInfo.hpp>
#include <optypes.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/defines.h>

#include <backend.hpp>
#include <cast.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>

using namespace detail;

static af_array cast(const af_array in, const af_dtype type) {
    const ArrayInfo& info = getInfo(in, false, true);

    if (info.getType() == type) { return retain(in); }

    if (info.isSparse()) {
        switch (type) {
            case f32: return getHandle(castSparse<float>(in));
            case f64: return getHandle(castSparse<double>(in));
            case c32: return getHandle(castSparse<cfloat>(in));
            case c64: return getHandle(castSparse<cdouble>(in));
            default: TYPE_ERROR(2, type);
        }
    } else {
        switch (type) {
            case f32: return getHandle(castArray<float>(in));
            case f64: return getHandle(castArray<double>(in));
            case c32: return getHandle(castArray<cfloat>(in));
            case c64: return getHandle(castArray<cdouble>(in));
            case s32: return getHandle(castArray<int>(in));
            case u32: return getHandle(castArray<uint>(in));
            case u8: return getHandle(castArray<uchar>(in));
            case b8: return getHandle(castArray<char>(in));
            case s64: return getHandle(castArray<intl>(in));
            case u64: return getHandle(castArray<uintl>(in));
            case s16: return getHandle(castArray<short>(in));
            case u16: return getHandle(castArray<ushort>(in));
            default: TYPE_ERROR(2, type);
        }
    }
}

af_err af_cast(af_array* out, const af_array in, const af_dtype type) {
    try {
        const ArrayInfo& info = getInfo(in, false, true);

        af_dtype inType = info.getType();
        if ((inType == c32 || inType == c64) && (type == f32 || type == f64)) {
            AF_ERROR(
                "Casting is not allowed from complex (c32/c64) to real "
                "(f32/f64) types.\n"
                "Use abs, real, imag etc to convert complex to floating type.",
                AF_ERR_TYPE);
        }

        dim4 idims = info.dims();
        if (idims.elements() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res = cast(in, type);

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
