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
#include <common/half.hpp>
#include <handle.hpp>
#include <transpose.hpp>
#include <af/arith.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline af_array trs(const af_array in, const bool conjugate) {
    return getHandle<T>(detail::transpose<T>(getArray<T>(in), conjugate));
}

af_err af_transpose(af_array* out, af_array in, const bool conjugate) {
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        if (dims.elements() == 0) { return af_retain_array(out, in); }

        if (dims[0] == 1 || dims[1] == 1) {
            af::dim4 outDims(dims[1], dims[0], dims[2], dims[3]);
            if (conjugate) {
                af_array temp = 0;
                AF_CHECK(af_conjg(&temp, in));
                AF_CHECK(af_moddims(out, temp, outDims.ndims(), outDims.get()));
                AF_CHECK(af_release_array(temp));
                return AF_SUCCESS;
            } else {
                // for a vector OR a batch of vectors
                // we can use modDims to transpose
                AF_CHECK(af_moddims(out, in, outDims.ndims(), outDims.get()));
                return AF_SUCCESS;
            }
        }

        af_array output;
        switch (type) {
            case f32: output = trs<float>(in, conjugate); break;
            case c32: output = trs<cfloat>(in, conjugate); break;
            case f64: output = trs<double>(in, conjugate); break;
            case c64: output = trs<cdouble>(in, conjugate); break;
            case b8: output = trs<char>(in, conjugate); break;
            case s32: output = trs<int>(in, conjugate); break;
            case u32: output = trs<uint>(in, conjugate); break;
            case u8: output = trs<uchar>(in, conjugate); break;
            case s64: output = trs<intl>(in, conjugate); break;
            case u64: output = trs<uintl>(in, conjugate); break;
            case s16: output = trs<short>(in, conjugate); break;
            case u16: output = trs<ushort>(in, conjugate); break;
            case f16: output = trs<half>(in, conjugate); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void transpose_inplace(af_array in, const bool conjugate) {
    return detail::transpose_inplace<T>(getArray<T>(in), conjugate);
}

af_err af_transpose_inplace(af_array in, const bool conjugate) {
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        // InPlace only works on square matrices
        DIM_ASSERT(0, dims[0] == dims[1]);

        // If singleton element
        if (dims[0] == 1) { return AF_SUCCESS; }

        switch (type) {
            case f32: transpose_inplace<float>(in, conjugate); break;
            case c32: transpose_inplace<cfloat>(in, conjugate); break;
            case f64: transpose_inplace<double>(in, conjugate); break;
            case c64: transpose_inplace<cdouble>(in, conjugate); break;
            case b8: transpose_inplace<char>(in, conjugate); break;
            case s32: transpose_inplace<int>(in, conjugate); break;
            case u32: transpose_inplace<uint>(in, conjugate); break;
            case u8: transpose_inplace<uchar>(in, conjugate); break;
            case s64: transpose_inplace<intl>(in, conjugate); break;
            case u64: transpose_inplace<uintl>(in, conjugate); break;
            case s16: transpose_inplace<short>(in, conjugate); break;
            case u16: transpose_inplace<ushort>(in, conjugate); break;
            case f16: transpose_inplace<half>(in, conjugate); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
