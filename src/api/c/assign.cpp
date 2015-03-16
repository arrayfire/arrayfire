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
#include <assign.hpp>
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

af_err af_assign_seq(af_array *out,
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
        else             res = weakCopy(lhs);

        try {

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

        } catch(...) {
            if (*out != lhs) {
                AF_CHECK(af_destroy_array(res));
            }
            throw;
        }
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static void genAssign(af_array& out, const af_index_t* indexers, const af_array& rhs)
{
    detail::assign<T>(getWritableArray<T>(out), indexers, getArray<T>(rhs));
}

af_err af_assign_gen(af_array *out,
                    const af_array lhs,
                    const dim_type ndims, const af_index_t* indexers,
                    const af_array rhs)
{
    af_array output = 0;
    // spanner is sequence indexer used for indexing along the
    // dimensions after ndims
    af_index_t spanner;
    spanner.mIndexer.seq = af_span;
    spanner.mIsSeq = true;

    try {
        ARG_ASSERT(2, (ndims>0));
        ARG_ASSERT(3, (indexers!=NULL));

        int track = 0;
        vector<af_seq> seqs(4, af_span);
        for (dim_type i = 0; i < ndims; i++) {
            if (indexers[i].mIsSeq) {
                track++;
                seqs[i] = indexers[i].mIndexer.seq;
            }
        }

        if (track==ndims) {
            // all indexers are sequences, redirecting to af_index
            return af_assign_seq(out, lhs, ndims, &(seqs.front()), rhs);
        }

        ARG_ASSERT(1, (lhs!=0));
        ARG_ASSERT(4, (rhs!=0));

        if (*out != lhs) AF_CHECK(af_copy_array(&output, lhs));
        else             output = lhs;

        ArrayInfo lInfo = getInfo(output);
        ArrayInfo rInfo = getInfo(rhs);
        dim4 lhsDims    = lInfo.dims();
        dim4 rhsDims    = rInfo.dims();
        af_dtype lhsType= lInfo.getType();
        af_dtype rhsType= rInfo.getType();

        ARG_ASSERT(1, (lhsType==rhsType));
        ARG_ASSERT(3, (rhsDims.ndims()>0));
        ARG_ASSERT(1, (lhsDims.ndims()>=rhsDims.ndims()));
        ARG_ASSERT(2, (lhsDims.ndims()>=(int)ndims));

        dim4 oDims = af::toDims(seqs, lhsDims);
        // if af_array are indexers along any
        // particular dimension, set the length of
        // that dimension accordingly before any checks
        for (dim_type i=0; i<ndims; i++) {
            if (!indexers[i].mIsSeq) {
                oDims[i] = getInfo(indexers[i].mIndexer.arr).elements();
            }
        }

        for (int i = 0; i < 4; i++) {
            if (oDims[i] != rhsDims[i])
                AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
        }

        af_index_t idxrs[4];
        // set all dimensions above ndims to spanner indexer
        for (dim_type i=ndims; i<4; ++i) idxrs[i] = spanner;

        for (dim_type i=0; i<ndims; ++i) {
            if (!indexers[i].mIsSeq) {
                // check if all af_arrays have atleast one value
                // to enable indexing along that dimension
                ArrayInfo idxInfo = getInfo(indexers[i].mIndexer.arr);
                af_dtype idxType  = idxInfo.getType();

                ARG_ASSERT(3, (idxType!=c32));
                ARG_ASSERT(3, (idxType!=c64));
                ARG_ASSERT(3, (idxType!=b8 ));

                idxrs[i].mIndexer.arr = indexers[i].mIndexer.arr;
                idxrs[i].mIsSeq = indexers[i].mIsSeq;
            } else {
                // af_seq is being used for this dimension
                // just copy the indexer to local variable
                idxrs[i] = indexers[i];
            }
        }

        try {
            switch(rhsType) {
                case c64: genAssign<cdouble>(output, idxrs, rhs); break;
                case f64: genAssign<double >(output, idxrs, rhs); break;
                case c32: genAssign<cfloat >(output, idxrs, rhs); break;
                case f32: genAssign<float  >(output, idxrs, rhs); break;
                case u64: genAssign<uintl  >(output, idxrs, rhs); break;
                case u32: genAssign<uint   >(output, idxrs, rhs); break;
                case s64: genAssign<intl   >(output, idxrs, rhs); break;
                case s32: genAssign<int    >(output, idxrs, rhs); break;
                case  u8: genAssign<uchar  >(output, idxrs, rhs); break;
                case  b8: genAssign<char   >(output, idxrs, rhs); break;
                default: TYPE_ERROR(1, rhsType);
            }
        } catch(...) {
            if (*out != lhs) AF_CHECK(af_destroy_array(output));
            throw;
        }
    }
    CATCHALL;

    std::swap(*out, output);

    return AF_SUCCESS;
}
