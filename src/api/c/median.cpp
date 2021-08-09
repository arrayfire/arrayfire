/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <sort.hpp>
#include <af/arith.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/index.h>
#include <af/statistics.h>

using af::dim4;
using detail::Array;
using detail::division;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::sort;

template<typename T>
static double median(const af_array& in) {
    dim_t nElems = getInfo(in).elements();
    dim4 dims(nElems, 1, 1, 1);
    ARG_ASSERT(0, nElems > 0);

    af_array temp = 0;
    AF_CHECK(af_moddims(&temp, in, 1, dims.get()));
    const Array<T>& input = getArray<T>(temp);

    // Shortcut cases for 1 or 2 elements
    if (nElems == 1) {
        T result;
        AF_CHECK(af_get_data_ptr((void*)&result, in));
        return result;
    }
    if (nElems == 2) {
        T result[2];
        AF_CHECK(af_get_data_ptr((void*)&result, in));
        return division(
            (static_cast<double>(result[0]) + static_cast<double>(result[1])),
            2.0);
    }

    double mid       = static_cast<double>(nElems + 1) / 2.0;
    af_seq mdSpan[1] = {af_make_seq(mid - 1, mid, 1)};

    Array<T> sortedArr = sort<T>(input, 0, true);

    af_array sarrHandle = getHandle<T>(sortedArr);

    double result;
    T resPtr[2];
    af_array res = 0;
    AF_CHECK(af_index(&res, sarrHandle, 1, mdSpan));
    AF_CHECK(af_get_data_ptr((void*)&resPtr, res));

    AF_CHECK(af_release_array(res));
    AF_CHECK(af_release_array(sarrHandle));
    AF_CHECK(af_release_array(temp));

    if (nElems % 2 == 1) {
        result = resPtr[0];
    } else {
        result = division(
            static_cast<double>(resPtr[0]) + static_cast<double>(resPtr[1]),
            2.0);
    }

    return result;
}

template<typename T>
static af_array median(const af_array& in, const dim_t dim) {
    const Array<T> input = getArray<T>(in);

    // Shortcut cases for 1 element along selected dimension
    if (input.dims()[dim] == 1) {
        Array<T> result = copyArray<T>(input);
        return getHandle<T>(result);
    }

    Array<T> sortedIn = sort<T>(input, dim, true);

    size_t dimLength = input.dims()[dim];
    double mid       = static_cast<double>(dimLength + 1) / 2.0;
    af_array left    = 0;

    af_seq slices[4] = {af_span, af_span, af_span, af_span};
    slices[dim]      = af_make_seq(mid - 1.0, mid - 1.0, 1.0);

    af_array sortedIn_handle = getHandle<T>(sortedIn);
    AF_CHECK(af_index(&left, sortedIn_handle, input.ndims(), slices));

    af_array out = nullptr;
    if (dimLength % 2 == 1) {
        // mid-1 is our guy
        if (input.isFloating()) {
            AF_CHECK(af_release_array(sortedIn_handle));
            return left;
        }

        // Return as floats for consistency
        af_array out;
        AF_CHECK(af_cast(&out, left, f32));
        AF_CHECK(af_release_array(left));
        AF_CHECK(af_release_array(sortedIn_handle));
        return out;
    } else {
        // ((mid-1)+mid)/2 is our guy
        dim4 dims      = input.dims();
        af_array right = 0;
        slices[dim]    = af_make_seq(mid, mid, 1.0);

        AF_CHECK(af_index(&right, sortedIn_handle, dims.ndims(), slices));

        af_array sumarr = 0;
        af_array carr   = 0;

        dim4 cdims = dims;
        cdims[dim] = 1;
        AF_CHECK(af_constant(&carr, 0.5, cdims.ndims(), cdims.get(),
                             input.isDouble() ? f64 : f32));

        if (!input.isFloating()) {
            af_array lleft, rright;
            AF_CHECK(af_cast(&lleft, left, f32));
            AF_CHECK(af_cast(&rright, right, f32));
            AF_CHECK(af_release_array(left));
            AF_CHECK(af_release_array(right));
            left  = lleft;
            right = rright;
        }

        AF_CHECK(af_add(&sumarr, left, right, false));
        AF_CHECK(af_mul(&out, sumarr, carr, false));

        AF_CHECK(af_release_array(left));
        AF_CHECK(af_release_array(right));
        AF_CHECK(af_release_array(sumarr));
        AF_CHECK(af_release_array(carr));
        AF_CHECK(af_release_array(sortedIn_handle));
    }
    return out;
}

af_err af_median_all(double* realVal, double* imagVal,  // NOLINT
                     const af_array in) {
    UNUSED(imagVal);
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();

        ARG_ASSERT(2, info.ndims() > 0);
        switch (type) {
            case f64: *realVal = median<double>(in); break;
            case f32: *realVal = median<float>(in); break;
            case s32: *realVal = median<int>(in); break;
            case u32: *realVal = median<uint>(in); break;
            case s16: *realVal = median<short>(in); break;
            case u16: *realVal = median<ushort>(in); break;
            case u8: *realVal = median<uchar>(in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_median(af_array* out, const af_array in, const dim_t dim) {
    try {
        ARG_ASSERT(2, (dim >= 0 && dim <= 4));

        af_array output       = 0;
        const ArrayInfo& info = getInfo(in);

        ARG_ASSERT(1, info.ndims() > 0);
        af_dtype type = info.getType();
        switch (type) {
            case f64: output = median<double>(in, dim); break;
            case f32: output = median<float>(in, dim); break;
            case s32: output = median<int>(in, dim); break;
            case u32: output = median<uint>(in, dim); break;
            case s16: output = median<short>(in, dim); break;
            case u16: output = median<ushort>(in, dim); break;
            case u8: output = median<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
