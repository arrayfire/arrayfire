/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <approx.hpp>

using af::dim4;
using namespace detail;

template<typename Ty, typename Tp>
static inline af_array approx1(const af_array in, const af_array pos,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(*approx1<Ty>(getArray<Ty>(in), getArray<Tp>(pos), method, offGrid));
}

template<typename Ty, typename Tp>
static inline af_array approx2(const af_array in, const af_array pos0, const af_array pos1,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(*approx2<Ty>(getArray<Ty>(in), getArray<Tp>(pos0), getArray<Tp>(pos1),
                                 method, offGrid));
}

af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                  const af_interp_type method, const float offGrid)
{
    try {
        ArrayInfo i_info = getInfo(in);
        ArrayInfo p_info = getInfo(pos);

        af_dtype itype = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());                       // Only floating and complex types
        ARG_ASSERT(2, p_info.isRealFloating());                   // Only floating types
        ARG_ASSERT(1, i_info.isSingle() == p_info.isSingle());    // Must have same precision
        ARG_ASSERT(1, i_info.isDouble() == p_info.isDouble());    // Must have same precision
        DIM_ASSERT(2, p_info.isColumn());                         // Only 1D input allowed
        ARG_ASSERT(3, (method == AF_INTERP_LINEAR || method == AF_INTERP_NEAREST));

        af_array output;

        switch(itype) {
            case f32: output = approx1<float  , float >(in, pos, method, offGrid);  break;
            case f64: output = approx1<double , double>(in, pos, method, offGrid);  break;
            case c32: output = approx1<cfloat , float >(in, pos, method, offGrid);  break;
            case c64: output = approx1<cdouble, double>(in, pos, method, offGrid);  break;
            default:  TYPE_ERROR(1, itype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                  const af_interp_type method, const float offGrid)
{
    try {
        ArrayInfo i_info = getInfo(in);
        ArrayInfo p_info = getInfo(pos0);
        ArrayInfo q_info = getInfo(pos1);

        af_dtype itype = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());                       // Only floating and complex types
        ARG_ASSERT(2, p_info.isRealFloating());                   // Only floating types
        ARG_ASSERT(3, q_info.isRealFloating());                   // Only floating types
        ARG_ASSERT(1, p_info.getType() == q_info.getType());      // Must have same type
        ARG_ASSERT(1, i_info.isSingle() == p_info.isSingle());    // Must have same precision
        ARG_ASSERT(1, i_info.isDouble() == p_info.isDouble());    // Must have same precision
        DIM_ASSERT(2, p_info.dims() == q_info.dims());            // POS0 and POS1 must have same dims
        DIM_ASSERT(2, p_info.ndims() < 3);// Allowing input batch but not positions. Output dims = (px, py, iz, iw)
        ARG_ASSERT(3, (method == AF_INTERP_LINEAR || method == AF_INTERP_NEAREST));

        af_array output;

        switch(itype) {
            case f32: output = approx2<float  , float >(in, pos0, pos1, method, offGrid);  break;
            case f64: output = approx2<double , double>(in, pos0, pos1, method, offGrid);  break;
            case c32: output = approx2<cfloat , float >(in, pos0, pos1, method, offGrid);  break;
            case c64: output = approx2<cdouble, double>(in, pos0, pos1, method, offGrid);  break;
            default:  TYPE_ERROR(1, itype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
