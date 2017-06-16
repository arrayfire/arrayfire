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
#include <af/arith.h>
#include <af/data.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <lookup.hpp>
#include <index.hpp>

using namespace detail;
using std::vector;
using std::swap;

template<typename T>
static void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index)
{
    const Array<T> &parent = getArray<T>(src);
    vector<af_seq> index_(index, index+ndims);
    Array<T> dst =  createSubArray(parent, index_);

    dest = getHandle(dst);
}

af_err af_index(af_array *result, const af_array in, const unsigned ndims, const af_seq* index)
{
    af_array out;
    try {

        const ArrayInfo& iInfo = getInfo(in);
        if (ndims == 1 && ndims != iInfo.ndims()) {
            af_array tmp_in;
            AF_CHECK(af_flat(&tmp_in, in));
            AF_CHECK(af_index(result, tmp_in, ndims, index));
            AF_CHECK(af_release_array(tmp_in));
            return AF_SUCCESS;
        }

        af_dtype in_type = iInfo.getType();

        switch(in_type) {
        case f32:    indexArray<float>   (out, in, ndims, index);  break;
        case c32:    indexArray<cfloat>  (out, in, ndims, index);  break;
        case f64:    indexArray<double>  (out, in, ndims, index);  break;
        case c64:    indexArray<cdouble> (out, in, ndims, index);  break;
        case b8:     indexArray<char>    (out, in, ndims, index);  break;
        case s32:    indexArray<int>     (out, in, ndims, index);  break;
        case u32:    indexArray<unsigned>(out, in, ndims, index);  break;
        case s16:    indexArray<short>   (out, in, ndims, index);  break;
        case u16:    indexArray<ushort>  (out, in, ndims, index);  break;
        case s64:    indexArray<intl>    (out, in, ndims, index);  break;
        case u64:    indexArray<uintl>   (out, in, ndims, index);  break;
        case u8:     indexArray<uchar>   (out, in, ndims, index);  break;
        default:    TYPE_ERROR(1, in_type);
        }
        swap(*result, out);
    }
    CATCHALL

    return AF_SUCCESS;
}

template<typename idx_t>
static af_array lookup(const af_array &in, const af_array &idx, const unsigned dim)
{
    const ArrayInfo& inInfo = getInfo(in);

    af_dtype inType  = inInfo.getType();

    switch(inType) {
        case f32: return getHandle(lookup<float   , idx_t > (getArray<float   >(in), getArray<idx_t>(idx), dim));
        case c32: return getHandle(lookup<cfloat  , idx_t > (getArray<cfloat  >(in), getArray<idx_t>(idx), dim));
        case f64: return getHandle(lookup<double  , idx_t > (getArray<double  >(in), getArray<idx_t>(idx), dim));
        case c64: return getHandle(lookup<cdouble , idx_t > (getArray<cdouble >(in), getArray<idx_t>(idx), dim));
        case s32: return getHandle(lookup<int     , idx_t > (getArray<int     >(in), getArray<idx_t>(idx), dim));
        case u32: return getHandle(lookup<unsigned, idx_t > (getArray<unsigned>(in), getArray<idx_t>(idx), dim));
        case s64: return getHandle(lookup<intl    , idx_t > (getArray<intl    >(in), getArray<idx_t>(idx), dim));
        case u64: return getHandle(lookup<uintl   , idx_t > (getArray<uintl   >(in), getArray<idx_t>(idx), dim));
        case s16: return getHandle(lookup<short   , idx_t > (getArray<short   >(in), getArray<idx_t>(idx), dim));
        case u16: return getHandle(lookup<ushort  , idx_t > (getArray<ushort  >(in), getArray<idx_t>(idx), dim));
        case  u8: return getHandle(lookup<uchar   , idx_t > (getArray<uchar   >(in), getArray<idx_t>(idx), dim));
        case  b8: return getHandle(lookup<char    , idx_t > (getArray<char    >(in), getArray<idx_t>(idx), dim));
        default : TYPE_ERROR(1, inType);
    }
}

af_err af_lookup(af_array *out, const af_array in, const af_array indices, const unsigned dim)
{
    af_array output = 0;

    try {
        ARG_ASSERT(3, (dim>=0 && dim<=3));

        const ArrayInfo& idxInfo= getInfo(indices);

        if(idxInfo.ndims() == 0) {
            return af_retain_array(out, indices);
        }

        ARG_ASSERT(2, idxInfo.isVector() || idxInfo.isScalar());

        af_dtype idxType = idxInfo.getType();

        ARG_ASSERT(2, (idxType!=c32));
        ARG_ASSERT(2, (idxType!=c64));
        ARG_ASSERT(2, (idxType!=b8));

        switch(idxType) {
            case f32: output = lookup<float   >(in, indices, dim); break;
            case f64: output = lookup<double  >(in, indices, dim); break;
            case s32: output = lookup<int     >(in, indices, dim); break;
            case u32: output = lookup<unsigned>(in, indices, dim); break;
            case s16: output = lookup<short   >(in, indices, dim); break;
            case u16: output = lookup<ushort  >(in, indices, dim); break;
            case s64: output = lookup<intl    >(in, indices, dim); break;
            case u64: output = lookup<uintl   >(in, indices, dim); break;
            case  u8: output = lookup<uchar   >(in, indices, dim); break;
            default : TYPE_ERROR(1, idxType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

// idxrs parameter to the below static function
// expects 4 values which is handled appropriately
// by the C-API af_index_gen
template<typename T>
static inline
af_array genIndex(const af_array& in,  const af_index_t idxrs[])
{
    return getHandle<T>(index<T>(getArray<T>(in), idxrs));
}

af_err af_index_gen(af_array *out, const af_array in, const dim_t ndims, const af_index_t* indexs)
{
    af_array output = 0;
    // spanner is sequence index used for indexing along the
    // dimensions after ndims
    af_index_t spanner;
    spanner.idx.seq = af_span;
    spanner.isSeq = true;

    try {
        ARG_ASSERT(2, (ndims>0));
        ARG_ASSERT(3, (indexs!=NULL));

        const ArrayInfo& iInfo = getInfo(in);

        dim4 iDims = iInfo.dims();
        af_dtype inType = getInfo(in).getType();

        if(iDims.ndims() <= 0) {
            return af_create_handle(out, 0, nullptr, inType);
        }

        if (ndims == 1 && ndims != (dim_t)iInfo.ndims()) {
            af_array tmp_in;
            AF_CHECK(af_flat(&tmp_in, in));
            AF_CHECK(af_index_gen(out, tmp_in, ndims, indexs));
            AF_CHECK(af_release_array(tmp_in));
            return AF_SUCCESS;
        }

        int track = 0;
        af_seq seqs[] = {af_span, af_span, af_span, af_span};
        for (dim_t i = 0; i < ndims; i++) {
            if (indexs[i].isSeq) {
                track++;
                seqs[i] = indexs[i].idx.seq;
            }
        }

        if (track==(int)ndims) {
            // all indexs are sequences, redirecting to af_index
            return af_index(out, in, ndims, seqs);
        }

        af_index_t idxrs[4];
        // set all dimensions above ndims to spanner index
        for (dim_t i=ndims; i<4; ++i) idxrs[i] = spanner;

        for (dim_t i=0; i<ndims; ++i) {
            if (!indexs[i].isSeq) {
                // check if all af_arrays have atleast one value
                // to enable indexing along that dimension
                const ArrayInfo& idxInfo = getInfo(indexs[i].idx.arr);
                af_dtype idxType  = idxInfo.getType();

                ARG_ASSERT(3, (idxType!=c32));
                ARG_ASSERT(3, (idxType!=c64));
                ARG_ASSERT(3, (idxType!=b8 ));

                idxrs[i].idx.arr = indexs[i].idx.arr;
                idxrs[i].isSeq = indexs[i].isSeq;
            } else {
                // af_seq is being used for this dimension
                // just copy the index to local variable
                idxrs[i] = indexs[i];
            }
        }

        switch(inType) {
            case c64: output = genIndex<cdouble>(in, idxrs); break;
            case f64: output = genIndex<double >(in, idxrs); break;
            case c32: output = genIndex<cfloat >(in, idxrs); break;
            case f32: output = genIndex<float  >(in, idxrs); break;
            case u64: output = genIndex<uintl  >(in, idxrs); break;
            case s64: output = genIndex<intl   >(in, idxrs); break;
            case u32: output = genIndex<uint   >(in, idxrs); break;
            case s32: output = genIndex<int    >(in, idxrs); break;
            case u16: output = genIndex<ushort >(in, idxrs); break;
            case s16: output = genIndex<short  >(in, idxrs); break;
            case  u8: output = genIndex<uchar  >(in, idxrs); break;
            case  b8: output = genIndex<char   >(in, idxrs); break;
            default: TYPE_ERROR(1, inType);
        }
    }
    CATCHALL;

    std::swap(*out, output);

    return AF_SUCCESS;
}

af_seq af_make_seq(double begin, double end, double step)
{
    af_seq seq = {begin, end, step};
    return seq;
}

af_err af_create_indexers(af_index_t** indexers)
{
    try {
        af_index_t* out = new af_index_t[4];
        for (int i=0; i<4; ++i) {
            out[i].idx.seq = af_span;
            out[i].isSeq = true;
            out[i].isBatch = false;
        }
        std::swap(*indexers, out);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_array_indexer(af_index_t* indexer, const af_array idx, const dim_t dim)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(1, (idx!=NULL));
        ARG_ASSERT(2, (dim>=0 && dim<=3));
        indexer[dim].idx.arr = idx;
        indexer[dim].isBatch = false;
        indexer[dim].isSeq   = false;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seq_indexer(af_index_t* indexer, const af_seq* idx, const dim_t dim, const bool is_batch)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(1, (idx!=NULL));
        ARG_ASSERT(2, (dim>=0 && dim<=3));
        indexer[dim].idx.seq = *idx;
        indexer[dim].isBatch = is_batch;
        indexer[dim].isSeq   = true;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seq_param_indexer(af_index_t* indexer,
                              const double begin, const double end, const double step,
                              const dim_t dim, const bool is_batch)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(4, (dim>=0 && dim<=3));
        indexer[dim].idx.seq = af_make_seq(begin, end, step);
        indexer[dim].isBatch = is_batch;
        indexer[dim].isSeq   = true;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_release_indexers(af_index_t* indexers)
{
    try {
        delete[] indexers;
    }
    CATCHALL;
    return AF_SUCCESS;
}
