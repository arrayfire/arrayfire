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

    dim4 oDims = af::toDims(index_, outDs);

    for (int i = 0; i < 4; i++) {
        if (oDims[i] != iDims[i])
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
    }


    Array<T> dst = createSubArray<T>(getArray<T>(out), index_, false);

    bool noCaseExecuted = true;
    if (isComplex) {
        noCaseExecuted = false;
        switch(iType) {
            case c64: copyArray<cdouble, T>(dst, getArray<cdouble>(in));  break;
            case c32: copyArray<cfloat , T>(dst, getArray<cfloat >(in));  break;
            default : noCaseExecuted = true; break;
        }
    }

    if(noCaseExecuted) {
        noCaseExecuted = false;
        switch(iType) {
            case f64: copyArray<double , T>(dst, getArray<double>(in));  break;
            case f32: copyArray<float  , T>(dst, getArray<float >(in));  break;
            case s32: copyArray<int    , T>(dst, getArray<int   >(in));  break;
            case u32: copyArray<uint   , T>(dst, getArray<uint  >(in));  break;
            case u8 : copyArray<uchar  , T>(dst, getArray<uchar >(in));  break;
            case b8 : copyArray<char   , T>(dst, getArray<char  >(in));  break;
            default : noCaseExecuted = true; break;
        }
    }

    if (noCaseExecuted)
        TYPE_ERROR(1, iType);
}

af_err af_assign(af_array *out,
                 const af_array lhs, const unsigned ndims,
                 const af_seq *index, const af_array rhs)
{
    try {
        ARG_ASSERT(0, (lhs!=0));
        ARG_ASSERT(1, (ndims>0));
        ARG_ASSERT(3, (rhs!=0));

        for(dim_type i=0; i<(dim_type)ndims; ++i) {
            ARG_ASSERT(2, (index[i].step>=0));
        }

        af_array res;
        if (*out != lhs) AF_CHECK(af_copy_array(&res, lhs));
        else             res = lhs;

        if (lhs != rhs) {
            ArrayInfo oInfo = getInfo(lhs);
            af_dtype oType  = oInfo.getType();
            switch(oType) {
            case c64: assign<cdouble, true >(res, ndims, index, rhs);  break;
            case c32: assign<cfloat , true >(res, ndims, index, rhs);  break;
            case f64: assign<double , false>(res, ndims, index, rhs);  break;
            case f32: assign<float  , false>(res, ndims, index, rhs);  break;
            case s32: assign<int    , false>(res, ndims, index, rhs);  break;
            case u32: assign<uint   , false>(res, ndims, index, rhs);  break;
            case u8 : assign<uchar  , false>(res, ndims, index, rhs);  break;
            case b8 : assign<char   , false>(res, ndims, index, rhs);  break;
            default : TYPE_ERROR(1, oType); break;
            }
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
