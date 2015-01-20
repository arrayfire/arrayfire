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
#include <af/index.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <math.hpp>

using namespace detail;
using std::vector;
using std::swap;

template<typename T, bool isComplex>
static
void assign(af_array &out, const unsigned &ndims, const af_seq *index, const af_array &in)
{
    ArrayInfo iInfo = getInfo(in);
    ArrayInfo oInfo = getInfo(out);
    af_dtype iType  = iInfo.getType();

    dim4 const outDs = oInfo.dims();
    dim4 const iDims = iInfo.dims();

    ARG_ASSERT(0, (outDs.ndims()>=iDims.ndims()));
    ARG_ASSERT(1, (outDs.ndims()>=(int)ndims));

    AF_CHECK(af_eval(out));

    vector<af_seq> index_(index, index+ndims);
    dim4 const oStrides = af::toStride(index_, outDs);

    dim4 oDims = af::toDims(index_, outDs);
    dim4 oOffsets = af::toOffset(index_, outDs);

    Array<T> *dst = createRefArray<T>(getArray<T>(out), oDims, oOffsets, oStrides);

    for (int i = 0; i < 4; i++) {
        if (oDims[i] != iDims[i])
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
    }

    bool noCaseExecuted = true;
    if (isComplex) {
        noCaseExecuted = false;
        switch(iType) {
            case c64: copy<cdouble, T>(*dst, getArray<cdouble>(in), scalar<T>(0), 1.0);  break;
            case c32: copy<cfloat , T>(*dst, getArray<cfloat >(in), scalar<T>(0), 1.0);  break;
            default : noCaseExecuted = true; break;
        }
    }

    if(noCaseExecuted) {
        noCaseExecuted = false;
        switch(iType) {
            case f64: copy<double , T>(*dst, getArray<double>(in), scalar<T>(0), 1.0);  break;
            case f32: copy<float  , T>(*dst, getArray<float >(in), scalar<T>(0), 1.0);  break;
            case s32: copy<int    , T>(*dst, getArray<int   >(in), scalar<T>(0), 1.0);  break;
            case u32: copy<uint   , T>(*dst, getArray<uint  >(in), scalar<T>(0), 1.0);  break;
            case u8 : copy<uchar  , T>(*dst, getArray<uchar >(in), scalar<T>(0), 1.0);  break;
            case b8 : copy<char   , T>(*dst, getArray<char  >(in), scalar<T>(0), 1.0);  break;
            default : noCaseExecuted = true; break;
        }
    }

    if (noCaseExecuted)
        TYPE_ERROR(1, iType);

    delete dst;
}

af_err af_assign(af_array out, const unsigned ndims, const af_seq *index, const af_array in)
{
    try {
        ARG_ASSERT(0, (out!=0));
        ARG_ASSERT(1, (ndims>0));
        ARG_ASSERT(3, (in!=0));

        if (in==out) {
            //FIXME: This should check for *index and throw exception if not equal
            return AF_SUCCESS;
        }

        for(dim_type i=0; i<(dim_type)ndims; ++i) {
            ARG_ASSERT(2, (index[i].begin>=0 || index[i].begin == -1));
            ARG_ASSERT(2, (index[i].step>=0));
        }

        ArrayInfo oInfo = getInfo(out);
        af_dtype oType  = oInfo.getType();
        switch(oType) {
            case c64: assign<cdouble, true >(out, ndims, index, in);  break;
            case c32: assign<cfloat , true >(out, ndims, index, in);  break;
            case f64: assign<double , false>(out, ndims, index, in);  break;
            case f32: assign<float  , false>(out, ndims, index, in);  break;
            case s32: assign<int    , false>(out, ndims, index, in);  break;
            case u32: assign<uint   , false>(out, ndims, index, in);  break;
            case u8 : assign<uchar  , false>(out, ndims, index, in);  break;
            case b8 : assign<char   , false>(out, ndims, index, in);  break;
            default : TYPE_ERROR(1, oType); break;
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
