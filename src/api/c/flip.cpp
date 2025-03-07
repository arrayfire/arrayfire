/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <handle.hpp>
#include <af/array.h>
#include <af/data.h>

#include <cassert>

using af::dim4;
using arrayfire::getArray;
using arrayfire::common::flip;
using arrayfire::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uintl;
using detail::ushort;
using std::swap;

template<typename T>
static inline af_array flip(const af_array in, const unsigned dim) {
    return getHandle(
        flip(getArray<T>(in), {dim == 0, dim == 1, dim == 2, dim == 3}));
}

af_err af_flip(af_array *result, const af_array in, const unsigned dim) {
    af_array out;
    try {
        const ArrayInfo &in_info = getInfo(in);

        if (in_info.ndims() <= dim) {
            *result = retain(in);
            return AF_SUCCESS;
        }

        af_dtype in_type = in_info.getType();

        switch (in_type) {
            case f16: out = flip<half>(in, dim); break;
            case f32: out = flip<float>(in, dim); break;
            case c32: out = flip<cfloat>(in, dim); break;
            case f64: out = flip<double>(in, dim); break;
            case c64: out = flip<cdouble>(in, dim); break;
            case b8: out = flip<char>(in, dim); break;
            case s32: out = flip<int>(in, dim); break;
            case u32: out = flip<unsigned>(in, dim); break;
            case s64: out = flip<intl>(in, dim); break;
            case u64: out = flip<uintl>(in, dim); break;
            case s16: out = flip<short>(in, dim); break;
            case u16: out = flip<ushort>(in, dim); break;
            case u8: out = flip<uchar>(in, dim); break;
            default: TYPE_ERROR(1, in_type);
        }
        swap(*result, out);
    }
    CATCHALL

    return AF_SUCCESS;
}
