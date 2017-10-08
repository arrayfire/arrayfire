/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/data.h>
#include <af/statistics.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <mean.hpp>
#include <arith.hpp>
#include <math.hpp>
#include <cast.hpp>

#include "stats.h"

using namespace detail;

template<typename Ti, typename To>
static To mean(const af_array &in)
{
    typedef typename baseOutType<To>::type Tw;
    return mean<Ti, Tw, To>(getArray<Ti>(in));
}

template<typename T>
static T mean(const af_array &in, const af_array &weights)
{
    typedef typename baseOutType<T>::type Tw;
    return mean<T, Tw>(castArray<T>(in), castArray<Tw>(weights));
}

template<typename Ti, typename To>
static af_array mean(const af_array &in, const dim_t dim)
{
    typedef typename baseOutType<To>::type Tw;
    return getHandle<To>(mean<Ti, Tw, To>(getArray<Ti>(in), dim));
}

template<typename T>
static af_array mean(const af_array &in, const af_array &weights, const dim_t dim)
{
    typedef typename baseOutType<T>::type Tw;
    return getHandle<T>(mean<T, Tw>(castArray<T>(in), castArray<Tw>(weights), dim));
}

af_err af_mean(af_array *out, const af_array in, const dim_t dim)
{
    try {
        ARG_ASSERT(2, (dim>=0 && dim<=3));

        af_array output = 0;
        const ArrayInfo& info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: output = mean<double  ,  double>(in, dim); break;
            case f32: output = mean<float   ,  float >(in, dim); break;
            case s32: output = mean<int     ,  float >(in, dim); break;
            case u32: output = mean<unsigned,  float >(in, dim); break;
            case s64: output = mean<intl    ,  double>(in, dim); break;
            case u64: output = mean<uintl   ,  double>(in, dim); break;
            case s16: output = mean<short   ,  float >(in, dim); break;
            case u16: output = mean<ushort  ,  float >(in, dim); break;
            case  u8: output = mean<uchar   ,  float >(in, dim); break;
            case  b8: output = mean<char    ,  float >(in, dim); break;
            case c32: output = mean<cfloat  ,  cfloat>(in, dim); break;
            case c64: output = mean<cdouble , cdouble>(in, dim); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_mean_weighted(af_array *out, const af_array in, const af_array weights, const dim_t dim)
{
    try {
        ARG_ASSERT(3, (dim>=0 && dim<=3));

        af_array output = 0;
        const ArrayInfo& iInfo = getInfo(in);
        const ArrayInfo& wInfo = getInfo(weights);
        af_dtype iType  = iInfo.getType();
        af_dtype wType  = wInfo.getType();

        ARG_ASSERT(2, (wType==f32 || wType==f64)); /* verify that weights are non-complex real numbers */

        //FIXME: We should avoid additional copies
        af_array w = weights;
        if (iInfo.dims() != wInfo.dims()) {
            dim4 iDims = iInfo.dims();
            dim4 wDims = wInfo.dims();
            dim4 tDims(1,1,1,1);
            for (int i = 0; i < 4; i++) {
                ARG_ASSERT(2, wDims[i] == 1 || wDims[i] == iDims[i]);
                tDims[i] = iDims[i] / wDims[i];
            }
            AF_CHECK(af_tile(&w, weights, tDims[0], tDims[1], tDims[2], tDims[3]));
        }

        switch(iType) {
            case f64: output = mean< double>(in, w, dim); break;
            case f32: output = mean< float >(in, w, dim); break;
            case s32: output = mean< float >(in, w, dim); break;
            case u32: output = mean< float >(in, w, dim); break;
            case s64: output = mean< double>(in, w, dim); break;
            case u64: output = mean< double>(in, w, dim); break;
            case s16: output = mean< float >(in, w, dim); break;
            case u16: output = mean< float >(in, w, dim); break;
            case  u8: output = mean< float >(in, w, dim); break;
            case  b8: output = mean< float >(in, w, dim); break;
            case c32: output = mean< cfloat>(in, w, dim); break;
            case c64: output = mean<cdouble>(in, w, dim); break;
            default : TYPE_ERROR(1, iType);
        }

        if (w != weights) {
            AF_CHECK(af_release_array(w));
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_mean_all(double *realVal, double *imagVal, const af_array in)
{
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = mean<double  , double>(in); break;
            case f32: *realVal = mean<float   , float >(in); break;
            case s32: *realVal = mean<int     , float >(in); break;
            case u32: *realVal = mean<unsigned, float >(in); break;
            case s64: *realVal = mean<intl    , double>(in); break;
            case u64: *realVal = mean<uintl   , double>(in); break;
            case s16: *realVal = mean<short   , float >(in); break;
            case u16: *realVal = mean<ushort  , float >(in); break;
            case  u8: *realVal = mean<uchar   , float >(in); break;
            case  b8: *realVal = mean<char    , float >(in); break;
            case c32: {
                cfloat tmp = mean<cfloat, cfloat>(in);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = mean<cdouble, cdouble>(in);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_mean_all_weighted(double *realVal, double *imagVal, const af_array in, const af_array weights)
{
    try {
        const ArrayInfo& iInfo = getInfo(in);
        const ArrayInfo& wInfo = getInfo(weights);
        af_dtype iType  = iInfo.getType();
        af_dtype wType  = wInfo.getType();

        ARG_ASSERT(3, (wType==f32 || wType==f64)); /* verify that weights are non-complex real numbers */

        switch(iType) {
            case f64: *realVal = mean<double>(in, weights); break;
            case f32: *realVal = mean< float>(in, weights); break;
            case s32: *realVal = mean< float>(in, weights); break;
            case u32: *realVal = mean< float>(in, weights); break;
            case s64: *realVal = mean<double>(in, weights); break;
            case u64: *realVal = mean<double>(in, weights); break;
            case s16: *realVal = mean< float>(in, weights); break;
            case u16: *realVal = mean< float>(in, weights); break;
            case  u8: *realVal = mean< float>(in, weights); break;
            case  b8: *realVal = mean< float>(in, weights); break;
            case c32: {
                cfloat tmp = mean<cfloat>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = mean<cdouble>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, iType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
