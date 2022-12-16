/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/moddims.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

namespace {
template<typename T>
af_array modDims(const af_array in, const dim4& newDims) {
    return getHandle(arrayfire::common::modDims(getArray<T>(in), newDims));
}
template<typename T>
af_array flat(const af_array in) {
    return getHandle(arrayfire::common::flat(getArray<T>(in)));
}
}  // namespace

af_err af_moddims(af_array* out, const af_array in, const unsigned ndims,
                  const dim_t* const dims) {
    try {
        if (ndims == 0) {
            *out = retain(in);
            return AF_SUCCESS;
        }
        ARG_ASSERT(2, ndims >= 1);
        ARG_ASSERT(3, dims != NULL);

        af_array output = 0;
        dim4 newDims(ndims, dims);
        const ArrayInfo& info = getInfo(in);
        dim_t in_elements     = info.elements();
        dim_t new_elements    = newDims.elements();

        DIM_ASSERT(1, in_elements == new_elements);

        af_dtype type = info.getType();

        switch (type) {
            case f32: output = modDims<float>(in, newDims); break;
            case c32: output = modDims<cfloat>(in, newDims); break;
            case f64: output = modDims<double>(in, newDims); break;
            case c64: output = modDims<cdouble>(in, newDims); break;
            case b8: output = modDims<char>(in, newDims); break;
            case s32: output = modDims<int>(in, newDims); break;
            case u32: output = modDims<uint>(in, newDims); break;
            case u8: output = modDims<uchar>(in, newDims); break;
            case s64: output = modDims<intl>(in, newDims); break;
            case u64: output = modDims<uintl>(in, newDims); break;
            case s16: output = modDims<short>(in, newDims); break;
            case u16: output = modDims<ushort>(in, newDims); break;
            case f16: output = modDims<half>(in, newDims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL

    return AF_SUCCESS;
}

af_err af_flat(af_array* out, const af_array in) {
    try {
        const ArrayInfo& info = getInfo(in);

        if (info.ndims() == 1) {
            *out = retain(in);
        } else {
            af_array output = 0;
            af_dtype type   = info.getType();

            switch (type) {
                case f32: output = flat<float>(in); break;
                case c32: output = flat<cfloat>(in); break;
                case f64: output = flat<double>(in); break;
                case c64: output = flat<cdouble>(in); break;
                case b8: output = flat<char>(in); break;
                case s32: output = flat<int>(in); break;
                case u32: output = flat<uint>(in); break;
                case u8: output = flat<uchar>(in); break;
                case s64: output = flat<intl>(in); break;
                case u64: output = flat<uintl>(in); break;
                case s16: output = flat<short>(in); break;
                case u16: output = flat<ushort>(in); break;
                case f16: output = flat<half>(in); break;
                default: TYPE_ERROR(1, type);
            }
            std::swap(*out, output);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
