/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <vector>
#include <cassert>

#include <af/array.h>
#include <af/index.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <ArrayIndex.hpp>

using namespace detail;
using std::vector;
using std::swap;

template<typename T>
static void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index)
{
    using af::toOffset;
    using af::toDims;
    using af::toStride;

    const Array<T> &parent = getArray<T>(src);
    vector<af_seq> index_(index, index+ndims);

    Array<T>* dst =  createSubArray(    parent,
                                        toDims(index_, parent.dims()),
                                        toOffset(index_, parent.dims()),
                                        toStride(index_, parent.dims()) );
    dest = getHandle(*dst);
}

af_err af_index(af_array *result, const af_array in, const unsigned ndims, const af_seq* index)
{
    af_array out;
    try {
        af_dtype in_type = getInfo(in).getType();

        switch(in_type) {
        case f32:    indexArray<float>   (out, in, ndims, index);  break;
        case c32:    indexArray<cfloat>  (out, in, ndims, index);  break;
        case f64:    indexArray<double>  (out, in, ndims, index);  break;
        case c64:    indexArray<cdouble> (out, in, ndims, index);  break;
        case b8:     indexArray<char>    (out, in, ndims, index);  break;
        case s32:    indexArray<int>     (out, in, ndims, index);  break;
        case u32:    indexArray<unsigned>(out, in, ndims, index);  break;
        case u8:     indexArray<uchar>   (out, in, ndims, index);  break;
        default:    TYPE_ERROR(1, in_type);
        }
    }
    CATCHALL

    swap(*result, out);
    return AF_SUCCESS;
}

template<typename idx_t>
static af_array arrayIndex(const af_array &in, const af_array &idx, const unsigned dim)
{
    ArrayInfo inInfo = getInfo(in);

    af_dtype inType  = inInfo.getType();

    switch(inType) {
        case f32: return getHandle(*arrayIndex<float   , idx_t > (getArray<float   >(in), getArray<idx_t>(idx), dim));
        case c32: return getHandle(*arrayIndex<cfloat  , idx_t > (getArray<cfloat  >(in), getArray<idx_t>(idx), dim));
        case f64: return getHandle(*arrayIndex<double  , idx_t > (getArray<double  >(in), getArray<idx_t>(idx), dim));
        case c64: return getHandle(*arrayIndex<cdouble , idx_t > (getArray<cdouble >(in), getArray<idx_t>(idx), dim));
        case s32: return getHandle(*arrayIndex<int     , idx_t > (getArray<int     >(in), getArray<idx_t>(idx), dim));
        case u32: return getHandle(*arrayIndex<unsigned, idx_t > (getArray<unsigned>(in), getArray<idx_t>(idx), dim));
        case  u8: return getHandle(*arrayIndex<uchar   , idx_t > (getArray<uchar   >(in), getArray<idx_t>(idx), dim));
        case  b8: return getHandle(*arrayIndex<char    , idx_t > (getArray<char    >(in), getArray<idx_t>(idx), dim));
        default : TYPE_ERROR(1, inType);
    }
}

af_err af_array_index(af_array *out, const af_array in, const af_array indices, const unsigned dim)
{
    af_array output = 0;

    try {
        ARG_ASSERT(3, (dim>=0 && dim<=3));

        ArrayInfo inInfo = getInfo(in);
        ArrayInfo idxInfo= getInfo(indices);

        ARG_ASSERT(2, (idxInfo.ndims()==1));

        af_dtype idxType = idxInfo.getType();

        ARG_ASSERT(2, (idxType!=c32));
        ARG_ASSERT(2, (idxType!=c64));
        ARG_ASSERT(2, (idxType!=b8));

        switch(idxType) {
            case f32: output = arrayIndex<float   >(in, indices, dim); break;
            case f64: output = arrayIndex<double  >(in, indices, dim); break;
            case s32: output = arrayIndex<int     >(in, indices, dim); break;
            case u32: output = arrayIndex<unsigned>(in, indices, dim); break;
            case  u8: output = arrayIndex<uchar   >(in, indices, dim); break;
            default : TYPE_ERROR(1, idxType);
        }
    }
    CATCHALL;

    std::swap(*out, output);

    return AF_SUCCESS;
}
