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
#include <diff.hpp>
#include <handle.hpp>
#include <af/algorithm.h>
#include <af/defines.h>

using af::dim4;
using arrayfire::getArray;
using arrayfire::getHandle;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline af_array diff1(const af_array in, const int dim) {
    return getHandle(diff1<T>(getArray<T>(in), dim));
}

template<typename T>
static inline af_array diff2(const af_array in, const int dim) {
    return getHandle(diff2<T>(getArray<T>(in), dim));
}

af_err af_diff1(af_array* out, const af_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();

        af::dim4 in_dims = info.dims();
        if (in_dims[dim] < 2) {
            return af_create_handle(out, 0, nullptr, type);
        }

        DIM_ASSERT(1, in_dims[dim] >= 2);

        af_array output;

        switch (type) {
            case f32: output = diff1<float>(in, dim); break;
            case c32: output = diff1<cfloat>(in, dim); break;
            case f64: output = diff1<double>(in, dim); break;
            case c64: output = diff1<cdouble>(in, dim); break;
            case b8: output = diff1<char>(in, dim); break;
            case s32: output = diff1<int>(in, dim); break;
            case u32: output = diff1<uint>(in, dim); break;
            case s64: output = diff1<intl>(in, dim); break;
            case u64: output = diff1<uintl>(in, dim); break;
            case s16: output = diff1<short>(in, dim); break;
            case u16: output = diff1<ushort>(in, dim); break;
            case u8: output = diff1<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_diff2(af_array* out, const af_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();

        af::dim4 in_dims = info.dims();
        if (in_dims[dim] < 3) {
            return af_create_handle(out, 0, nullptr, type);
        }
        DIM_ASSERT(1, in_dims[dim] >= 3);

        af_array output;

        switch (type) {
            case f32: output = diff2<float>(in, dim); break;
            case c32: output = diff2<cfloat>(in, dim); break;
            case f64: output = diff2<double>(in, dim); break;
            case c64: output = diff2<cdouble>(in, dim); break;
            case b8: output = diff2<char>(in, dim); break;
            case s32: output = diff2<int>(in, dim); break;
            case u32: output = diff2<uint>(in, dim); break;
            case s64: output = diff2<intl>(in, dim); break;
            case u64: output = diff2<uintl>(in, dim); break;
            case s16: output = diff2<short>(in, dim); break;
            case u16: output = diff2<ushort>(in, dim); break;
            case u8: output = diff2<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
