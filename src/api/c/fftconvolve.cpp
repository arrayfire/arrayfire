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
#include <fftconvolve.hpp>
#include <convolve_common.hpp>

#include <cstdio>

using af::dim4;
using namespace detail;

template<typename T, typename convT, bool isDouble, bool roundOut, dim_type baseDim>
inline static af_array fftconvolve(const af_array &s, const af_array &f, ConvolveBatchKind kind)
{
    return getHandle(fftconvolve<T, convT, isDouble, roundOut, baseDim>(getArray<T>(s), getArray<T>(f), kind));
}

template<dim_type baseDim>
ConvolveBatchKind identifyBatchKind(const dim4 &sDims, const dim4 &fDims)
{
    dim_type sn = sDims.ndims();
    dim_type fn = fDims.ndims();

    if (sn==baseDim && fn==baseDim)
        return ONE2ONE;
    else if (sn==baseDim && (fn>baseDim && fn<=4))
        return ONE2MANY;
    else if ((sn>baseDim && sn<=4) && fn==baseDim)
        return MANY2ONE;
    else if ((sn>baseDim && sn<=4) && (fn>baseDim && fn<=4)) {
        bool doesDimensionsMatch = true;
        for (dim_type i=baseDim; i<4; i++) {
            if (sDims[i]!=fDims[i]) {
                doesDimensionsMatch = false;
                break;
            }
        }
        return (doesDimensionsMatch ? MANY2MANY : CONVOLVE_UNSUPPORTED_BATCH_MODE);
    }
    else
        return CONVOLVE_UNSUPPORTED_BATCH_MODE;
}

template<dim_type baseDim>
af_err fftconvolve(af_array *out, af_array signal, af_array filter)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo fInfo = getInfo(filter);

        af_dtype stype  = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        ConvolveBatchKind convBT = identifyBatchKind<baseDim>(sdims, fdims);

        ARG_ASSERT(1, (convBT != CONVOLVE_UNSUPPORTED_BATCH_MODE));

        af_array output;
        switch(stype) {
            case f64: output = fftconvolve<double, double, true , false, baseDim>(signal, filter, convBT); break;
            case f32: output = fftconvolve<float , float,  false, false, baseDim>(signal, filter, convBT); break;
            case u32: output = fftconvolve<uint  , float,  false, true,  baseDim>(signal, filter, convBT); break;
            case s32: output = fftconvolve<int   , float,  false, true,  baseDim>(signal, filter, convBT); break;
            case u8:  output = fftconvolve<uchar , float,  false, true,  baseDim>(signal, filter, convBT); break;
            case b8:  output = fftconvolve<char  , float,  false, true,  baseDim>(signal, filter, convBT); break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fftconvolve1(af_array *out, af_array signal, af_array filter)
{
    return fftconvolve<1>(out, signal, filter);
}

af_err af_fftconvolve2(af_array *out, af_array signal, af_array filter)
{
    return fftconvolve<2>(out, signal, filter);
}

af_err af_fftconvolve3(af_array *out, af_array signal, af_array filter)
{
    return fftconvolve<3>(out, signal, filter);
}
