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
#include <handle.hpp>
#include <shift.hpp>
#include <af/data.h>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline af_array shift(const af_array in, const int sdims[4]) {
    return getHandle(shift<T>(getArray<T>(in), sdims));
}

af_err af_shift(af_array *out, const af_array in, const int sdims[4]) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (info.ndims() == 0) { return af_retain_array(out, in); }
        DIM_ASSERT(1, info.elements() > 0);

        af_array output;

        switch (type) {
            case f32: output = shift<float>(in, sdims); break;
            case c32: output = shift<cfloat>(in, sdims); break;
            case f64: output = shift<double>(in, sdims); break;
            case c64: output = shift<cdouble>(in, sdims); break;
            case b8: output = shift<char>(in, sdims); break;
            case s32: output = shift<int>(in, sdims); break;
            case u32: output = shift<uint>(in, sdims); break;
            case s64: output = shift<intl>(in, sdims); break;
            case u64: output = shift<uintl>(in, sdims); break;
            case s16: output = shift<short>(in, sdims); break;
            case u16: output = shift<ushort>(in, sdims); break;
            case u8: output = shift<uchar>(in, sdims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_shift(af_array *out, const af_array in, const int x, const int y,
                const int z, const int w) {
    const int sdims[] = {x, y, z, w};
    return af_shift(out, in, sdims);
}
