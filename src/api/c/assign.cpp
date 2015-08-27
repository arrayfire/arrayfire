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
#include <af/data.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <assign.hpp>
#include <math.hpp>
#include <tile.hpp>

using namespace detail;
using std::vector;
using std::swap;

// From src/api/c/moddims.cpp TODO: move to header?
template<typename T>
Array<T> modDims(const Array<T>& in, const af::dim4 &newDims);

template<typename Tout, typename Tin>
static
void assign(Array<Tout> &out, const unsigned &ndims, const af_seq *index, const Array<Tin> &in_)
{
    dim4 const outDs = out.dims();
    dim4 const iDims = in_.dims();

    DIM_ASSERT(0, (outDs.ndims()>=iDims.ndims()));
    DIM_ASSERT(0, (outDs.ndims()>=(dim_t)ndims));

    evalArray(out);

    vector<af_seq> index_(index, index+ndims);

    dim4 oDims = toDims(index_, outDs);

    bool is_vector = true;
    for (int i = 0; is_vector && i < (int)oDims.ndims() - 1; i++) {
        is_vector &= oDims[i] == 1;
    }

    is_vector &= in_.isVector() || in_.isScalar();

    for (dim_t i = ndims; i < (int)in_.ndims(); i++) {
        oDims[i] = 1;
    }


    if (is_vector) {
        if (oDims.elements() != (dim_t)in_.elements() &&
            in_.elements() != 1) {
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
        }

        // If both out and in are vectors of equal elements, reshape in to out dims
        Array<Tin> in = in_.elements() == 1 ? tile(in_, oDims) : modDims(in_, oDims);
        Array<Tout> dst = createSubArray<Tout>(out, index_, false);

        copyArray<Tin , Tout>(dst, in);
    } else {
        for (int i = 0; i < 4; i++) {
            if (oDims[i] != iDims[i]) {
                AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
            }
        }
        Array<Tout> dst = createSubArray<Tout>(out, index_, false);

        copyArray<Tin , Tout>(dst, in_);
    }
}

template<typename T>
static
void assign_helper(Array<T> &out, const unsigned &ndims, const af_seq *index, const af_array &in_)
{
    ArrayInfo iInfo = getInfo(in_);
    af_dtype iType  = iInfo.getType();

    if(out.getType() == c64 || out.getType() == c32)
    {

        switch(iType) {
            case c64: assign<T, cdouble>(out, ndims, index, getArray<cdouble  >(in_));  break;
            case c32: assign<T, cfloat >(out, ndims, index, getArray<cfloat   >(in_));  break;
            default : TYPE_ERROR(1, iType); break;
        }
    }
    else
    {
        switch(iType) {
            case f64: assign<T, double >(out, ndims, index, getArray<double   >(in_));  break;
            case f32: assign<T, float  >(out, ndims, index, getArray<float    >(in_));  break;
            case s32: assign<T, int    >(out, ndims, index, getArray<int      >(in_));  break;
            case u32: assign<T, uint   >(out, ndims, index, getArray<uint     >(in_));  break;
            case s64: assign<T, intl   >(out, ndims, index, getArray<intl     >(in_));  break;
            case u64: assign<T, uintl  >(out, ndims, index, getArray<uintl    >(in_));  break;
            case u8 : assign<T, uchar  >(out, ndims, index, getArray<uchar    >(in_));  break;
            case b8 : assign<T, char   >(out, ndims, index, getArray<char     >(in_));  break;
            default : TYPE_ERROR(1, iType); break;
        }
    }
}

af_err af_assign_seq(af_array *out,
                     const af_array lhs, const unsigned ndims,
                     const af_seq *index, const af_array rhs)
{
    try {
        ARG_ASSERT(0, (lhs!=0));
        ARG_ASSERT(1, (ndims>0));
        ARG_ASSERT(3, (rhs!=0));

        ArrayInfo lInfo = getInfo(lhs);

        if (ndims == 1 && ndims != (dim_t)lInfo.ndims()) {
            af_array tmp_in, tmp_out;
            AF_CHECK(af_flat(&tmp_in, lhs));
            AF_CHECK(af_assign_seq(&tmp_out, tmp_in, ndims, index, rhs));
            AF_CHECK(af_moddims(out, tmp_out, lInfo.ndims(), lInfo.dims().get()));
            AF_CHECK(af_release_array(tmp_in));
            AF_CHECK(af_release_array(tmp_out));
            return AF_SUCCESS;
        }

        for(dim_t i=0; i<(dim_t)ndims; ++i) {
            ARG_ASSERT(2, (index[i].step>=0));
        }

        af_array res = 0;

        if (*out != lhs) {
            int count = 0;
            AF_CHECK(af_get_data_ref_count(&count, lhs));
            if (count > 1) {
                AF_CHECK(af_copy_array(&res, lhs));
            } else {
                AF_CHECK(af_retain_array(&res, lhs));
            }
        } else {
            res = lhs;
        }

        try {

            if (lhs != rhs) {
                ArrayInfo oInfo = getInfo(lhs);
                af_dtype oType  = oInfo.getType();
                switch(oType) {
                case c64: assign_helper<cdouble>(getWritableArray<cdouble>(res), ndims, index, rhs);  break;
                case c32: assign_helper<cfloat >(getWritableArray<cfloat >(res), ndims, index, rhs);  break;
                case f64: assign_helper<double >(getWritableArray<double >(res), ndims, index, rhs);  break;
                case f32: assign_helper<float  >(getWritableArray<float  >(res), ndims, index, rhs);  break;
                case s32: assign_helper<int    >(getWritableArray<int    >(res), ndims, index, rhs);  break;
                case u32: assign_helper<uint   >(getWritableArray<uint   >(res), ndims, index, rhs);  break;
                case s64: assign_helper<intl   >(getWritableArray<intl   >(res), ndims, index, rhs);  break;
                case u64: assign_helper<uintl  >(getWritableArray<uintl  >(res), ndims, index, rhs);  break;
                case u8 : assign_helper<uchar  >(getWritableArray<uchar  >(res), ndims, index, rhs);  break;
                case b8 : assign_helper<char   >(getWritableArray<char   >(res), ndims, index, rhs);  break;
                default : TYPE_ERROR(1, oType); break;
                }
            }
        } catch(...) {
            af_release_array(res);
            throw;
        }
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static void genAssign(af_array& out, const af_index_t* indexs, const af_array& rhs)
{
    detail::assign<T>(getWritableArray<T>(out), indexs, getArray<T>(rhs));
}

af_err af_assign_gen(af_array *out,
                    const af_array lhs,
                    const dim_t ndims, const af_index_t* indexs,
                    const af_array rhs_)
{
    af_array output = 0;
    af_array rhs = rhs_;
    // spanner is sequence index used for indexing along the
    // dimensions after ndims
    af_index_t spanner;
    spanner.idx.seq = af_span;
    spanner.isSeq = true;

    try {
        ARG_ASSERT(2, (ndims>0));
        ARG_ASSERT(3, (indexs!=NULL));

        int track = 0;
        vector<af_seq> seqs(4, af_span);
        for (dim_t i = 0; i < ndims; i++) {
            if (indexs[i].isSeq) {
                track++;
                seqs[i] = indexs[i].idx.seq;
            }
        }

        if (track==(int)ndims) {
            // all indexs are sequences, redirecting to af_assign
            return af_assign_seq(out, lhs, ndims, &(seqs.front()), rhs);
        }

        ARG_ASSERT(1, (lhs!=0));
        ARG_ASSERT(4, (rhs!=0));

        ArrayInfo lInfo = getInfo(lhs);
        ArrayInfo rInfo = getInfo(rhs);
        dim4 lhsDims    = lInfo.dims();
        dim4 rhsDims    = rInfo.dims();
        af_dtype lhsType= lInfo.getType();
        af_dtype rhsType= rInfo.getType();

        ARG_ASSERT(2, (ndims == 1) || (ndims == (dim_t)lInfo.ndims()));

        if (ndims == 1 && ndims != (dim_t)lInfo.ndims()) {
            af_array tmp_in, tmp_out;
            AF_CHECK(af_flat(&tmp_in, lhs));
            AF_CHECK(af_assign_gen(&tmp_out, tmp_in, ndims, indexs, rhs_));
            AF_CHECK(af_moddims(out, tmp_out, lInfo.ndims(), lInfo.dims().get()));
            AF_CHECK(af_release_array(tmp_in));
            AF_CHECK(af_release_array(tmp_out));
            return AF_SUCCESS;
        }

        ARG_ASSERT(1, (lhsType==rhsType));
        ARG_ASSERT(3, (rhsDims.ndims()>0));
        ARG_ASSERT(1, (lhsDims.ndims()>=rhsDims.ndims()));
        ARG_ASSERT(2, (lhsDims.ndims()>=ndims));

        if (*out != lhs) {
            int count = 0;
            AF_CHECK(af_get_data_ref_count(&count, lhs));
            if (count > 1) {
                AF_CHECK(af_copy_array(&output, lhs));
            } else {
                AF_CHECK(af_retain_array(&output, lhs));
            }
        } else {
            output = lhs;
        }

        dim4 oDims = toDims(seqs, lhsDims);
        // if af_array are indexs along any
        // particular dimension, set the length of
        // that dimension accordingly before any checks
        for (dim_t i=0; i<ndims; i++) {
            if (!indexs[i].isSeq) {
                oDims[i] = getInfo(indexs[i].idx.arr).elements();
            }
        }

        for (dim_t i = ndims; i < (dim_t)lInfo.ndims(); i++) {
            oDims[i] = 1;
        }

        bool is_vector = true;
        for (int i = 0; is_vector && i < oDims.ndims() - 1; i++) {
            is_vector &= oDims[i] == 1;
        }

        //TODO: Move logic out of this
        is_vector &= rInfo.isVector() || rInfo.isScalar();
        if (is_vector) {
            if (oDims.elements() != (dim_t)rInfo.elements() &&
                rInfo.elements() != 1) {
                AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
            }

            if (rInfo.elements() == 1) {
                AF_CHECK(af_tile(&rhs, rhs_, oDims[0], oDims[1], oDims[2], oDims[3]));
            } else {
                // If both out and rhs are vectors of equal elements, reshape rhs to out dims
                AF_CHECK(af_moddims(&rhs, rhs_, oDims.ndims(), oDims.get()));
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (oDims[i] != rhsDims[i]) {
                    AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
                }
            }
        }

        af_index_t idxrs[4];
        // set all dimensions above ndims to spanner index
        for (dim_t i=ndims; i<4; ++i) idxrs[i] = spanner;

        for (dim_t i=0; i<ndims; ++i) {
            if (!indexs[i].isSeq) {
                // check if all af_arrays have atleast one value
                // to enable indexing along that dimension
                ArrayInfo idxInfo = getInfo(indexs[i].idx.arr);
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
            if (*out != lhs) {
                AF_CHECK(af_release_array(output));
                if (is_vector) { AF_CHECK(af_release_array(rhs)); }
            }
            throw;
        }
        if (is_vector) { AF_CHECK(af_release_array(rhs)); }
    }
    CATCHALL;

    std::swap(*out, output);

    return AF_SUCCESS;
}
