/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/index.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <moments.hpp>
#include <handle.hpp>
#include <reorder.hpp>
#include <tile.hpp>
#include <join.hpp>
#include <cast.hpp>
#include <arith.hpp>

#include <iostream>
#include <limits>

using af::dim4;
using namespace detail;

template<typename T>
static inline void moments(af_array *out, const af_array in, af_moment_type moment)
{
    Array<float> temp = moments<T>(getArray<T>(in), moment);
    *out = getHandle<float>(temp);
}

af_err af_moments(af_array *out, const af_array in, const af_moment_type moment)
{
    try {
        const ArrayInfo& in_info = getInfo(in);
        af_dtype type = in_info.getType();

        switch(type) {
            case f32: moments<float>          (out, in, moment); break;
            case f64: moments<double>         (out, in, moment); break;
            case u32: moments<unsigned>       (out, in, moment); break;
            case s32: moments<int>            (out, in, moment); break;
            case u16: moments<unsigned short> (out, in, moment); break;
            case s16: moments<short>          (out, in, moment); break;
            case b8:  moments<char>           (out, in, moment); break;
            default:  TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void moment_copy(double* out, const af_array moments)
{
    dim_t elems;
    af_get_elements(&elems, moments);
    T *h_ptr = new T[elems];
    af_get_data_ptr((void *)h_ptr, moments);

    for(unsigned i=0; i<elems; ++i)
        out[i] = (double)h_ptr[i];
    delete[] h_ptr;
}

af_err af_moments_all(double* out, const af_array in, const af_moment_type moment)
{
    try {
        const ArrayInfo& in_info = getInfo(in);
        dim4 idims = in_info.dims();
        DIM_ASSERT(1, idims[2] == 1 && idims[3] == 1);

        af_array moments_arr;
        af_moments(&moments_arr, in, moment);
        moment_copy<float>(out, moments_arr);
    }
    CATCHALL;

    return AF_SUCCESS;
}
