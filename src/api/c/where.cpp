/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <where.hpp>
#include <af/algorithm.h>
#include <af/dim4.hpp>
#include <complex>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::swap;

template<typename T>
static inline af_array where(const af_array in) {
    // Making it more explicit that the output is uint
    return getHandle<uint>(where<T>(getArray<T>(in)));
}

af_err af_where(af_array* idx, const af_array in) {
    try {
        const ArrayInfo& i_info = getInfo(in);
        af_dtype type           = i_info.getType();

        if (i_info.ndims() == 0) {
            return af_create_handle(idx, 0, nullptr, u32);
        }

        af_array res;
        switch (type) {
            case f32: res = where<float>(in); break;
            case f64: res = where<double>(in); break;
            case c32: res = where<cfloat>(in); break;
            case c64: res = where<cdouble>(in); break;
            case s32: res = where<int>(in); break;
            case u32: res = where<uint>(in); break;
            case s64: res = where<intl>(in); break;
            case u64: res = where<uintl>(in); break;
            case s16: res = where<short>(in); break;
            case u16: res = where<ushort>(in); break;
            case u8: res = where<uchar>(in); break;
            case b8: res = where<char>(in); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*idx, res);
    }
    CATCHALL

    return AF_SUCCESS;
}
