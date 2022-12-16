/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <index.hpp>
#include <indexing_common.hpp>

#include <Array.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/moddims.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/index.h>

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

using std::signbit;
using std::swap;
using std::vector;

using af::dim4;
using arrayfire::common::convert2Canonical;
using arrayfire::common::createSpanIndex;
using arrayfire::common::flat;
using arrayfire::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::index;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

namespace arrayfire {
namespace common {
af_index_t createSpanIndex() {
    static af_index_t s = [] {
        af_index_t s;
        s.idx.seq = af_span;
        s.isSeq   = true;
        s.isBatch = false;
        return s;
    }();
    return s;
}

af_seq convert2Canonical(const af_seq s, const dim_t len) {
    double begin = signbit(s.begin) ? (len + s.begin) : s.begin;
    double end   = signbit(s.end) ? (len + s.end) : s.end;

    return af_seq{begin, end, s.step};
}
}  // namespace common
}  // namespace arrayfire

template<typename T>
static af_array indexBySeqs(const af_array& src,
                            const vector<af_seq>& indicesV) {
    auto ndims        = static_cast<dim_t>(indicesV.size());
    const auto& input = getArray<T>(src);

    if (ndims == 1U && ndims != input.ndims()) {
        return getHandle(createSubArray(flat(input), indicesV));
    } else {
        return getHandle(createSubArray(input, indicesV));
    }
}

af_err af_index(af_array* result, const af_array in, const unsigned ndims,
                const af_seq* indices) {
    try {
        ARG_ASSERT(2, (ndims > 0 && ndims <= AF_MAX_DIMS));

        const ArrayInfo& inInfo = getInfo(in);
        af_dtype type           = inInfo.getType();
        const dim4& iDims       = inInfo.dims();

        vector<af_seq> indices_(ndims, af_span);
        for (unsigned i = 0; i < ndims; ++i) {
            indices_[i] = convert2Canonical(indices[i], iDims[i]);

            ARG_ASSERT(3, (indices_[i].begin >= 0. && indices_[i].end >= 0.));
            if (signbit(indices_[i].step)) {
                ARG_ASSERT(3, indices_[i].begin >= indices_[i].end);
            } else {
                ARG_ASSERT(3, indices_[i].begin <= indices_[i].end);
            }
        }

        af_array out = 0;

        switch (type) {
            case f32: out = indexBySeqs<float>(in, indices_); break;
            case c32: out = indexBySeqs<cfloat>(in, indices_); break;
            case f64: out = indexBySeqs<double>(in, indices_); break;
            case c64: out = indexBySeqs<cdouble>(in, indices_); break;
            case b8: out = indexBySeqs<char>(in, indices_); break;
            case s32: out = indexBySeqs<int>(in, indices_); break;
            case u32: out = indexBySeqs<unsigned>(in, indices_); break;
            case s16: out = indexBySeqs<short>(in, indices_); break;
            case u16: out = indexBySeqs<ushort>(in, indices_); break;
            case s64: out = indexBySeqs<intl>(in, indices_); break;
            case u64: out = indexBySeqs<uintl>(in, indices_); break;
            case u8: out = indexBySeqs<uchar>(in, indices_); break;
            case f16: out = indexBySeqs<half>(in, indices_); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T, typename idx_t>
inline af_array lookup(const af_array& in, const af_array& idx,
                       const unsigned dim) {
    return getHandle(lookup(getArray<T>(in), getArray<idx_t>(idx), dim));
}

template<typename idx_t>
static af_array lookup(const af_array& in, const af_array& idx,
                       const unsigned dim) {
    const ArrayInfo& inInfo = getInfo(in);
    af_dtype inType         = inInfo.getType();

    switch (inType) {
        case f32: return lookup<float, idx_t>(in, idx, dim);
        case c32: return lookup<cfloat, idx_t>(in, idx, dim);
        case f64: return lookup<double, idx_t>(in, idx, dim);
        case c64: return lookup<cdouble, idx_t>(in, idx, dim);
        case s32: return lookup<int, idx_t>(in, idx, dim);
        case u32: return lookup<unsigned, idx_t>(in, idx, dim);
        case s64: return lookup<intl, idx_t>(in, idx, dim);
        case u64: return lookup<uintl, idx_t>(in, idx, dim);
        case s16: return lookup<short, idx_t>(in, idx, dim);
        case u16: return lookup<ushort, idx_t>(in, idx, dim);
        case u8: return lookup<uchar, idx_t>(in, idx, dim);
        case b8: return lookup<char, idx_t>(in, idx, dim);
        case f16: return lookup<half, idx_t>(in, idx, dim);
        default: TYPE_ERROR(1, inType);
    }
}

af_err af_lookup(af_array* out, const af_array in, const af_array indices,
                 const unsigned dim) {
    try {
        const ArrayInfo& idxInfo = getInfo(indices);

        if (idxInfo.ndims() == 0) {
            *out = retain(indices);
            return AF_SUCCESS;
        }

        ARG_ASSERT(3, (dim <= 3));
        ARG_ASSERT(2, idxInfo.isVector() || idxInfo.isScalar());

        af_dtype idxType = idxInfo.getType();

        ARG_ASSERT(2, (idxType != c32));
        ARG_ASSERT(2, (idxType != c64));
        ARG_ASSERT(2, (idxType != b8));

        af_array output = 0;

        switch (idxType) {
            case f32: output = lookup<float>(in, indices, dim); break;
            case f64: output = lookup<double>(in, indices, dim); break;
            case s32: output = lookup<int>(in, indices, dim); break;
            case u32: output = lookup<unsigned>(in, indices, dim); break;
            case s16: output = lookup<short>(in, indices, dim); break;
            case u16: output = lookup<ushort>(in, indices, dim); break;
            case s64: output = lookup<intl>(in, indices, dim); break;
            case u64: output = lookup<uintl>(in, indices, dim); break;
            case u8: output = lookup<uchar>(in, indices, dim); break;
            case f16: output = lookup<half>(in, indices, dim); break;
            default: TYPE_ERROR(1, idxType);
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
static inline af_array genIndex(const af_array& in, const af_index_t idxrs[]) {
    return getHandle<T>(index<T>(getArray<T>(in), idxrs));
}

af_err af_index_gen(af_array* out, const af_array in, const dim_t ndims,
                    const af_index_t* indexs) {
    try {
        ARG_ASSERT(2, (ndims > 0 && ndims <= AF_MAX_DIMS));
        ARG_ASSERT(3, (indexs != NULL));

        const ArrayInfo& iInfo = getInfo(in);
        const dim4& iDims      = iInfo.dims();
        af_dtype inType        = getInfo(in).getType();

        if (iDims.ndims() <= 0) {
            *out = createHandle(dim4(0), inType);
            return AF_SUCCESS;
        }

        if (ndims == 1 && ndims != static_cast<dim_t>(iInfo.ndims())) {
            af_array in_ = 0;
            AF_CHECK(af_flat(&in_, in));
            AF_CHECK(af_index_gen(out, in_, ndims, indexs));
            AF_CHECK(af_release_array(in_));
            return AF_SUCCESS;
        }

        int track = 0;
        std::array<af_seq, AF_MAX_DIMS> seqs{};
        seqs.fill(af_span);
        for (dim_t i = 0; i < ndims; i++) {
            if (indexs[i].isSeq) {
                track++;
                seqs[i] = indexs[i].idx.seq;
            }
        }

        if (track == static_cast<int>(ndims)) {
            return af_index(out, in, ndims, seqs.data());
        }

        std::array<af_index_t, AF_MAX_DIMS> idxrs{};

        for (dim_t i = 0; i < AF_MAX_DIMS; ++i) {
            if (i < ndims) {
                bool isSeq = indexs[i].isSeq;
                if (!isSeq) {
                    // check if all af_arrays have atleast one value
                    // to enable indexing along that dimension
                    const ArrayInfo& idxInfo = getInfo(indexs[i].idx.arr);
                    af_dtype idxType         = idxInfo.getType();

                    ARG_ASSERT(3, (idxType != c32));
                    ARG_ASSERT(3, (idxType != c64));
                    ARG_ASSERT(3, (idxType != b8));

                    idxrs[i] = {{indexs[i].idx.arr}, isSeq, indexs[i].isBatch};
                } else {
                    // copy the af_seq to local variable
                    af_seq inSeq =
                        convert2Canonical(indexs[i].idx.seq, iDims[i]);
                    ARG_ASSERT(3, (inSeq.begin >= 0. || inSeq.end >= 0.));
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

        af_array output = 0;
        switch (inType) {
            case c64: output = genIndex<cdouble>(in, ptr); break;
            case f64: output = genIndex<double>(in, ptr); break;
            case c32: output = genIndex<cfloat>(in, ptr); break;
            case f32: output = genIndex<float>(in, ptr); break;
            case u64: output = genIndex<uintl>(in, ptr); break;
            case s64: output = genIndex<intl>(in, ptr); break;
            case u32: output = genIndex<uint>(in, ptr); break;
            case s32: output = genIndex<int>(in, ptr); break;
            case u16: output = genIndex<ushort>(in, ptr); break;
            case s16: output = genIndex<short>(in, ptr); break;
            case u8: output = genIndex<uchar>(in, ptr); break;
            case b8: output = genIndex<char>(in, ptr); break;
            case f16: output = genIndex<half>(in, ptr); break;
            default: TYPE_ERROR(1, inType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_seq af_make_seq(double begin, double end, double step) {
    return af_seq{begin, end, step};
}

af_err af_create_indexers(af_index_t** indexers) {
    try {
        auto* out = new af_index_t[AF_MAX_DIMS];
        for (int i = 0; i < AF_MAX_DIMS; ++i) {
            out[i].idx.seq = af_span;
            out[i].isSeq   = true;
            out[i].isBatch = false;
        }
        std::swap(*indexers, out);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_array_indexer(af_index_t* indexer, const af_array idx,
                            const dim_t dim) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(1, (idx != NULL));
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));
        indexer[dim] = af_index_t{{idx}, false, false};
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_seq_indexer(af_index_t* indexer, const af_seq* idx,
                          const dim_t dim, const bool is_batch) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(1, (idx != NULL));
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));
        indexer[dim].idx.seq = *idx;
        indexer[dim].isSeq   = true;
        indexer[dim].isBatch = is_batch;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_seq_param_indexer(af_index_t* indexer, const double begin,
                                const double end, const double step,
                                const dim_t dim, const bool is_batch) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(4, (dim >= 0 && dim <= 3));
        af_seq s             = af_make_seq(begin, end, step);
        indexer[dim].idx.seq = s;
        indexer[dim].isSeq   = true;
        indexer[dim].isBatch = is_batch;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_release_indexers(af_index_t* indexers) {
    try {
        delete[] indexers;
    }
    CATCHALL;
    return AF_SUCCESS;
}
