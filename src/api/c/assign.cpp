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
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <assign.hpp>
#include <math.hpp>
#include <tile.hpp>
#include <indexing_common.hpp>

using namespace detail;
using std::vector;
using std::swap;
using std::signbit;
using common::convert2Canonical;
using common::createSpanIndex;

template<typename Tout, typename Tin>
static
void assign(Array<Tout> &out, const vector<af_seq> seqs,
            const Array<Tin> &in)
{
    size_t ndims  = seqs.size();
    const dim4& outDs = out.dims();
    const dim4& iDims = in.dims();

    if (iDims.elements() == 0) return;

    out.eval();

    dim4 oDims = toDims(seqs, outDs);

    bool isVec = true;
    for (int i = 0; isVec && i < (int)oDims.ndims() - 1; i++) {
        isVec &= oDims[i] == 1;
    }

    isVec &= in.isVector() || in.isScalar();

    for (dim_t i = ndims; i < (int)in.ndims(); i++) {
        oDims[i] = 1;
    }

    if (isVec) {
        if (oDims.elements() != (dim_t)in.elements() &&
            in.elements() != 1) {
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
        }

        // If both out and in are vectors of equal elements,
        // reshape in to out dims
        Array<Tin> in_ = in.elements() == 1 ? tile(in, oDims)
                                           : modDims(in, oDims);
        auto dst = createSubArray<Tout>(out, seqs, false);

        copyArray<Tin , Tout>(dst, in_);
    } else {
        for (int i = 0; i < AF_MAX_DIMS; i++) {
            if (oDims[i] != iDims[i])
                AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
        }
        Array<Tout> dst = createSubArray<Tout>(out, seqs, false);

        copyArray<Tin , Tout>(dst, in);
    }
}

template<typename T>
static
void assign(Array<T> &out, const vector<af_seq> iv,
            const af_array &in)
{
    const ArrayInfo& iInfo = getInfo(in);
    af_dtype iType = iInfo.getType();

    if(out.getType() == c64 || out.getType() == c32) {
        switch(iType) {
        case c64: assign<T, cdouble>(out, iv, getArray<cdouble>(in));  break;
        case c32: assign<T, cfloat >(out, iv, getArray<cfloat >(in));  break;
        default : TYPE_ERROR(1, iType); break;
        }
    } else {
        switch(iType) {
        case f64: assign<T, double >(out, iv, getArray<double >(in));  break;
        case f32: assign<T, float  >(out, iv, getArray<float  >(in));  break;
        case s32: assign<T, int    >(out, iv, getArray<int    >(in));  break;
        case u32: assign<T, uint   >(out, iv, getArray<uint   >(in));  break;
        case s64: assign<T, intl   >(out, iv, getArray<intl   >(in));  break;
        case u64: assign<T, uintl  >(out, iv, getArray<uintl  >(in));  break;
        case s16: assign<T, short  >(out, iv, getArray<short  >(in));  break;
        case u16: assign<T, ushort >(out, iv, getArray<ushort >(in));  break;
        case u8 : assign<T, uchar  >(out, iv, getArray<uchar  >(in));  break;
        case b8 : assign<T, char   >(out, iv, getArray<char   >(in));  break;
        default : TYPE_ERROR(1, iType); break;
        }
    }
}

af_err af_assign_seq(af_array *out,
                     const af_array lhs, const unsigned ndims,
                     const af_seq *index, const af_array rhs)
{
    try {
        ARG_ASSERT(0, (lhs != 0));
        ARG_ASSERT(1, (ndims > 0));
        ARG_ASSERT(3, (rhs != 0));

        const ArrayInfo& lInfo = getInfo(lhs);

        if (ndims == 1 && ndims != lInfo.ndims()) {
            af_array tmp_in, tmp_out;
            AF_CHECK(af_flat(&tmp_in, lhs));
            AF_CHECK(af_assign_seq(&tmp_out, tmp_in, ndims, index, rhs));
            AF_CHECK(af_moddims(out, tmp_out, lInfo.ndims(), lInfo.dims().get()));
            AF_CHECK(af_release_array(tmp_in));
            // This can run into a double free issue if tmp_in == tmp_out
            // The condition ensures release only if both are different
            // Issue found on Tegra X1
            if(tmp_in != tmp_out) AF_CHECK(af_release_array(tmp_out));
            return AF_SUCCESS;
        }

        af_array res = 0;

        if (*out != lhs) {
            int count = 0;
            AF_CHECK(af_get_data_ref_count(&count, lhs));
            if (count > 1)
                AF_CHECK(af_copy_array(&res, lhs));
            else
                res = retain(lhs);
        } else {
            res = lhs;
        }

        try {
            if (lhs != rhs) {
                const dim4& outDims = getInfo(res).dims();
                const dim4& inDims = getInfo(rhs).dims();

                vector<af_seq> inSeqs(ndims, af_span);
                for (unsigned i=0; i<ndims; ++i) {
                    inSeqs[i] = convert2Canonical(index[i], outDims[i]);
                    ARG_ASSERT(3, (inSeqs[i].begin >= 0. || inSeqs[i].end >= 0.));
                    if (signbit(inSeqs[i].step)) {
                        ARG_ASSERT(3, inSeqs[i].begin >= inSeqs[i].end);
                    } else {
                        ARG_ASSERT(3, inSeqs[i].begin <= inSeqs[i].end);
                    }
                }
                DIM_ASSERT(0, (outDims.ndims()>=inDims.ndims()));
                DIM_ASSERT(0, (outDims.ndims()>=(dim_t)ndims));

                const ArrayInfo& oInfo = getInfo(res);
                af_dtype oType = oInfo.getType();
                switch(oType) {
                case c64: assign(getWritableArray<cdouble>(res), inSeqs, rhs); break;
                case c32: assign(getWritableArray<cfloat >(res), inSeqs, rhs); break;
                case f64: assign(getWritableArray<double >(res), inSeqs, rhs); break;
                case f32: assign(getWritableArray<float  >(res), inSeqs, rhs); break;
                case s32: assign(getWritableArray<int    >(res), inSeqs, rhs); break;
                case u32: assign(getWritableArray<uint   >(res), inSeqs, rhs); break;
                case s64: assign(getWritableArray<intl   >(res), inSeqs, rhs); break;
                case u64: assign(getWritableArray<uintl  >(res), inSeqs, rhs); break;
                case s16: assign(getWritableArray<short  >(res), inSeqs, rhs); break;
                case u16: assign(getWritableArray<ushort >(res), inSeqs, rhs); break;
                case u8 : assign(getWritableArray<uchar  >(res), inSeqs, rhs); break;
                case b8 : assign(getWritableArray<char   >(res), inSeqs, rhs); break;
                default : TYPE_ERROR(1, oType); break;
                }
            }
        } catch(...) {
            af_release_array(res);
            throw;
        }
        swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
inline
void genAssign(af_array& out, const af_index_t* indexs, const af_array& rhs)
{
    detail::assign<T>(getWritableArray<T>(out), indexs, getArray<T>(rhs));
}

af_err af_assign_gen(af_array *out, const af_array lhs,
                    const dim_t ndims, const af_index_t* indexs,
                    const af_array rhs_)
{
    try {
        ARG_ASSERT(3, (indexs!=NULL));

        int track = 0;
        vector<af_seq> seqs(AF_MAX_DIMS, af_span);
        for (dim_t i = 0; i < ndims; i++) {
            if (indexs[i].isSeq) {
                track++;
                seqs[i] = indexs[i].idx.seq;
            }
        }

        af_array rhs = rhs_;
        if (track==(int)ndims) {
            // all indexs are sequences, redirecting to af_assign
            return af_assign_seq(out, lhs, ndims, seqs.data(), rhs);
        }

        ARG_ASSERT(1, (lhs!=0));
        ARG_ASSERT(4, (rhs!=0));

        const ArrayInfo& lInfo = getInfo(lhs);
        const ArrayInfo& rInfo = getInfo(rhs);
        const dim4& lhsDims    = lInfo.dims();
        const dim4& rhsDims    = rInfo.dims();
        af_dtype lhsType       = lInfo.getType();
        af_dtype rhsType       = rInfo.getType();

        if(rhsDims.ndims() == 0)
            return af_retain_array(out, lhs);

        if(lhsDims.ndims() == 0)
            return af_create_handle(out, 0, nullptr, lhsType);

        ARG_ASSERT(2, (ndims == 1) || (ndims == (dim_t)lInfo.ndims()));

        if (ndims == 1 && ndims != (dim_t)lInfo.ndims()) {
            af_array tmp_in = 0, tmp_out = 0;
            AF_CHECK(af_flat(&tmp_in, lhs));
            AF_CHECK(af_assign_gen(&tmp_out, tmp_in, ndims, indexs, rhs_));
            AF_CHECK(af_moddims(out, tmp_out, lInfo.ndims(), lInfo.dims().get()));
            AF_CHECK(af_release_array(tmp_in));
            // This can run into a double free issue if tmp_in == tmp_out
            // The condition ensures release only if both are different
            // Issue found on Tegra X1
            if(tmp_in != tmp_out) AF_CHECK(af_release_array(tmp_out));
            return AF_SUCCESS;
        }

        ARG_ASSERT(1, (lhsType==rhsType));
        ARG_ASSERT(1, (lhsDims.ndims()>=rhsDims.ndims()));
        ARG_ASSERT(2, (lhsDims.ndims()>=ndims));

        af_array output = 0;
        if (*out != lhs) {
            int count = 0;
            AF_CHECK(af_get_data_ref_count(&count, lhs));
            if (count > 1)
                AF_CHECK(af_copy_array(&output, lhs));
            else
                output = retain(lhs);
        } else {
            output = lhs;
        }

        dim4 oDims = toDims(seqs, lhsDims);
        // if af_array are indexs along any
        // particular dimension, set the length of
        // that dimension accordingly before any checks
        for (dim_t i=0; i<ndims; i++) {
            if (!indexs[i].isSeq)
                oDims[i] = getInfo(indexs[i].idx.arr).elements();
        }

        for (dim_t i = ndims; i < (dim_t)lInfo.ndims(); i++)
            oDims[i] = 1;

        bool isVec = true;
        for (int i = 0; isVec && i < oDims.ndims() - 1; i++) {
            isVec &= oDims[i] == 1;
        }

        //TODO: Move logic out of this
        isVec &= rInfo.isVector() || rInfo.isScalar();
        if (isVec) {
            if (oDims.elements() != (dim_t)rInfo.elements() &&
                rInfo.elements() != 1) {
                AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
            }
            if (rInfo.elements() == 1) {
                AF_CHECK(af_tile(&rhs, rhs_, oDims[0],
                                 oDims[1], oDims[2], oDims[3]));
            } else {
                // If both out and rhs are vectors of equal
                // elements, reshape rhs to out dims
                AF_CHECK(af_moddims(&rhs, rhs_, oDims.ndims(), oDims.get()));
            }
        } else {
            for (int i = 0; i < AF_MAX_DIMS; i++) {
                if (oDims[i] != rhsDims[i])
                    AF_ERROR("Size mismatch between input and output",
                             AF_ERR_SIZE);
            }
        }

        std::array<af_index_t, AF_MAX_DIMS> idxrs;
        for (dim_t i=0; i<AF_MAX_DIMS; ++i) {
            if (i < ndims) {
                bool isSeq = indexs[i].isSeq;
                if (!isSeq) {
                    // check if all af_arrays have atleast one value
                    // to enable indexing along that dimension
                    const ArrayInfo& idxInfo = getInfo(indexs[i].idx.arr);
                    af_dtype idxType = idxInfo.getType();

                    ARG_ASSERT(3, (idxType!=c32));
                    ARG_ASSERT(3, (idxType!=c64));
                    ARG_ASSERT(3, (idxType!=b8 ));

                    idxrs[i] = { { indexs[i].idx.arr },
                                 isSeq, indexs[i].isBatch };
                } else {
                    af_seq inSeq = convert2Canonical(indexs[i].idx.seq, lhsDims[i]);
                    ARG_ASSERT(3, (inSeq.begin >= 0 || inSeq.end >= 0));
                    if (signbit(inSeq.step)) {
                        ARG_ASSERT(3, inSeq.begin >= inSeq.end);
                    } else {
                        ARG_ASSERT(3, inSeq.begin <= inSeq.end);
                    }

                    idxrs[i].idx.seq = inSeq;
                    idxrs[i].isSeq   = isSeq;
                    idxrs[i].isBatch = indexs[i].isBatch;
                }
            } else {
                // set all dimensions above ndims to spanner
                idxrs[i] = createSpanIndex();
            }
        }
        af_index_t* ptr = idxrs.data();

        try {
            switch(rhsType) {
                case c64: genAssign<cdouble>(output, ptr, rhs); break;
                case f64: genAssign<double >(output, ptr, rhs); break;
                case c32: genAssign<cfloat >(output, ptr, rhs); break;
                case f32: genAssign<float  >(output, ptr, rhs); break;
                case u64: genAssign<uintl  >(output, ptr, rhs); break;
                case u32: genAssign<uint   >(output, ptr, rhs); break;
                case s64: genAssign<intl   >(output, ptr, rhs); break;
                case s32: genAssign<int    >(output, ptr, rhs); break;
                case s16: genAssign<short  >(output, ptr, rhs); break;
                case u16: genAssign<ushort >(output, ptr, rhs); break;
                case  u8: genAssign<uchar  >(output, ptr, rhs); break;
                case  b8: genAssign<char   >(output, ptr, rhs); break;
                default: TYPE_ERROR(1, rhsType);
            }
        } catch(...) {
            if (*out != lhs) {
                AF_CHECK(af_release_array(output));
                if (isVec)
                    AF_CHECK(af_release_array(rhs));
            }
            throw;
        }
        if (isVec)
            AF_CHECK(af_release_array(rhs));
        swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
