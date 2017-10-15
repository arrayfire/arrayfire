/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/image.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <histogram.hpp>

using af::dim4;
using namespace detail;

template<typename inType,typename outType>
static inline af_array histogram(const af_array in, const unsigned &nbins,
                                 const double &minval, const double &maxval,
                                 const bool islinear)
{
    if (islinear)
        return getHandle(histogram<inType,outType, true>(getArray<inType>(in),nbins,minval,maxval));
    else
        return getHandle(histogram<inType,outType, false>(getArray<inType>(in),nbins,minval,maxval));
}

af_err af_histogram(af_array *out, const af_array in,
                    const unsigned nbins, const double minval, const double maxval)
{
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type  = info.getType();

        if(info.ndims() == 0) {
            return af_retain_array(out, in);
        }

        af_array output;
        switch(type) {
            case f32: output = histogram<float , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case f64: output = histogram<double, uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case b8 : output = histogram<char  , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case s32: output = histogram<int   , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case u32: output = histogram<uint  , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case s16: output = histogram<short , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case u16: output = histogram<ushort, uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case s64: output = histogram<intl  , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case u64: output = histogram<uintl , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            case u8 : output = histogram<uchar , uint>(in, nbins, minval, maxval, info.isLinear()); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
