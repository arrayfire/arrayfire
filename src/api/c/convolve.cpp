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
#include <af/signal.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <convolve.hpp>
#include <convolve_common.hpp>

#include <cstdio>

using af::dim4;
using namespace detail;

template<typename T, typename accT, dim_type baseDim, bool expand>
inline static af_array convolve(const af_array &s, const af_array &f, ConvolveBatchKind kind)
{
    return getHandle(convolve<T, accT, baseDim, expand>(getArray<T>(s), castArray<accT>(f), kind));
}

template<typename T, typename accT, bool expand>
inline static af_array convolve2(const af_array &s, const af_array &c_f, const af_array &r_f)
{
    return getHandle(convolve2<T, accT, expand>(getArray<T>(s),
                                                castArray<accT>(c_f),
                                                castArray<accT>(r_f)));
}

template<dim_type baseDim>
ConvolveBatchKind identifyBatchKind(dim_type sn, dim_type fn)
{
    const dim_type batchDim = baseDim + 1;
    if (sn==baseDim && fn==batchDim)
        return ONE2ALL;
    else if (sn==batchDim && fn==batchDim)
        return MANY2MANY;
    else if (sn==batchDim && fn==baseDim)
        return MANY2ONE;
    else
        return ONE2ONE;
}

template<dim_type baseDim, bool expand>
af_err convolve(af_array *out, af_array signal, af_array filter)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo fInfo = getInfo(filter);

        af_dtype stype  = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        dim_type batchDim = baseDim+1;
        ARG_ASSERT(1, (sdims.ndims()<=batchDim));
        ARG_ASSERT(2, (fdims.ndims()<=batchDim));

        ConvolveBatchKind convBT = identifyBatchKind<baseDim>(sdims.ndims(), fdims.ndims());

        af_array output;
        switch(stype) {
            case c32: output = convolve<cfloat ,  cfloat, baseDim, expand>(signal, filter, convBT); break;
            case c64: output = convolve<cdouble, cdouble, baseDim, expand>(signal, filter, convBT); break;
            case f32: output = convolve<float  ,   float, baseDim, expand>(signal, filter, convBT); break;
            case f64: output = convolve<double ,  double, baseDim, expand>(signal, filter, convBT); break;
            case u32: output = convolve<uint   ,   float, baseDim, expand>(signal, filter, convBT); break;
            case s32: output = convolve<int    ,   float, baseDim, expand>(signal, filter, convBT); break;
            case u8:  output = convolve<uchar  ,   float, baseDim, expand>(signal, filter, convBT); break;
            case b8:  output = convolve<char   ,   float, baseDim, expand>(signal, filter, convBT); break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<bool expand>
af_err convolve2_sep(af_array *out, af_array col_filter, af_array row_filter, af_array signal)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo cfInfo= getInfo(col_filter);
        ArrayInfo rfInfo= getInfo(row_filter);

        af_dtype signalType  = sInfo.getType();

        dim4 signalDims = sInfo.dims();

        ARG_ASSERT(1, (signalDims.ndims()==2 || signalDims.ndims()==3));
        ARG_ASSERT(2, cfInfo.isVector());
        ARG_ASSERT(3, rfInfo.isVector());

        af_array output;
        switch(signalType) {
            case c32: output = convolve2<cfloat ,  cfloat, expand>(signal, col_filter, row_filter); break;
            case c64: output = convolve2<cdouble, cdouble, expand>(signal, col_filter, row_filter); break;
            case f32: output = convolve2<float  ,   float, expand>(signal, col_filter, row_filter); break;
            case f64: output = convolve2<double ,  double, expand>(signal, col_filter, row_filter); break;
            case u32: output = convolve2<uint   ,   float, expand>(signal, col_filter, row_filter); break;
            case s32: output = convolve2<int    ,   float, expand>(signal, col_filter, row_filter); break;
            case u8:  output = convolve2<uchar  ,   float, expand>(signal, col_filter, row_filter); break;
            case b8:  output = convolve2<char   ,   float, expand>(signal, col_filter, row_filter); break;
            default: TYPE_ERROR(1, signalType);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_convolve1(af_array *out, af_array signal, af_array filter, bool expand)
{
    if (expand)
        return convolve<1, true >(out, signal, filter);
    else
        return convolve<1, false>(out, signal, filter);
}

af_err af_convolve2(af_array *out, af_array signal, af_array filter, bool expand)
{
    if (expand)
        return convolve<2, true >(out, signal, filter);
    else
        return convolve<2, false>(out, signal, filter);
}

af_err af_convolve3(af_array *out, af_array signal, af_array filter, bool expand)
{
    if (expand)
        return convolve<3, true >(out, signal, filter);
    else
        return convolve<3, false>(out, signal, filter);
}

af_err af_convolve2_sep(af_array *out, af_array signal, af_array col_filter, af_array row_filter, bool expand)
{
    if (expand)
        return convolve2_sep<true >(out, signal, col_filter, row_filter);
    else
        return convolve2_sep<false>(out, signal, col_filter, row_filter);
}
