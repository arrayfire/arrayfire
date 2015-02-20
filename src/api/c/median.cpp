/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/statistics.h>
#include <af/index.h>
#include <af/arith.h>
#include <af/data.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <sort.hpp>
#include <math.hpp>
#include <cast.hpp>

using namespace detail;
using af::dim4;

template<typename T>
static T median(const af_array& in)
{
    const Array<T> input  = getArray<T>(in);
    dim_type nElems = input.elements();
    double mid      = (nElems + 1) / 2;
    af_seq mdSpan[1]= {mid-1, mid, 1};
    dim4 dims(nElems, 1, 1, 1);

    af_array temp = 0;
    AF_CHECK(af_moddims(&temp, in, 1, dims.get()));
    Array<T>* sortedArr = detail::sort<T, true>(input, 0);

    T result;
    T resPtr[2];
    af_array res = 0;
    AF_CHECK(af_index(&res, getHandle<T>(*sortedArr), 1, mdSpan));
    AF_CHECK(af_get_data_ptr((void*)&resPtr, res));

    if (nElems % 2 == 1) {
        result = *resPtr;
    } else {
        result = division(*resPtr + *(resPtr+1), 2);
    }

    AF_CHECK(af_destroy_array(res));
    destroyArray<T>(*sortedArr);
    AF_CHECK(af_destroy_array(temp));

    return result;
}

template<typename T>
static af_array median(const af_array& in, dim_type dim)
{
    const Array<T> input = getArray<T>(in);
    Array<T>* sortedIn   = detail::sort<T, true>(input, dim);

    int nElems    = input.elements();
    double mid    = (nElems + 1) / 2;
    af_array left = 0;

    af_seq slices[4] = {af_span, af_span, af_span, af_span};
    slices[dim] = {mid-1, mid-1, 1};

    AF_CHECK(af_index(&left, getHandle<T>(*sortedIn), input.ndims(), slices));

    if (nElems % 2 == 1) {
        // mid-1 is our guy
        destroyArray<T>(*sortedIn);
        return left;
    } else {
        // ((mid-1)+mid)/2 is our guy
        dim4  dims = input.dims();
        af_array right = 0;
        slices[dim] = {mid, mid, 1};

        AF_CHECK(af_index(&right, getHandle<T>(*sortedIn), dims.ndims(), slices));

        af_array sumarr = 0;
        af_array carr   = 0;
        af_array result = 0;

        AF_CHECK(af_constant(&carr, 0.5, dims.ndims(), dims.get(), input.getType()));
        AF_CHECK(af_add(&sumarr, left, right, false));
        AF_CHECK(af_mul(&result, sumarr, carr, false));

        AF_CHECK(af_destroy_array(left));
        AF_CHECK(af_destroy_array(right));
        AF_CHECK(af_destroy_array(sumarr));
        AF_CHECK(af_destroy_array(carr));
        destroyArray<T>(*sortedIn);

        return result;
    }
}

af_err af_median_all(double *realVal, double *imagVal, const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = median<double>(in); break;
            case f32: *realVal = median<float >(in); break;
            case s32: *realVal = median<int   >(in); break;
            case u32: *realVal = median<uint  >(in); break;
            case  u8: *realVal = median<uchar >(in); break;
            case  b8: *realVal = median<char  >(in); break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_median(af_array* out, const af_array in, dim_type dim)
{
    try {
        ARG_ASSERT(2, (dim>=0 && dim<=0));

        af_array output = 0;
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: output = median<double>(in, dim); break;
            case f32: output = median<float >(in, dim); break;
            case s32: output = median<int   >(in, dim); break;
            case u32: output = median<uint  >(in, dim); break;
            case  u8: output = median<uchar >(in, dim); break;
            case  b8: output = median<char  >(in, dim); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
