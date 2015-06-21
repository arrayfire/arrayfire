/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <limits>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <err_common.hpp>

namespace af
{
#if __cplusplus > 199711l
    static_assert(std::is_standard_layout<dim4>::value, "af::dim4 must be a standard layout type");
#endif

using std::vector;
using std::numeric_limits;
using std::abs;

dim4::dim4()
{
    dims[0] = 0;
    dims[1] = 0;
    dims[2] = 0;
    dims[3] = 0;
}

dim4::dim4( dim_t first,
            dim_t second,
            dim_t third,
            dim_t fourth)
{
    dims[0] = first;
    dims[1] = second;
    dims[2] = third;
    dims[3] = fourth;
}

dim4::dim4(const dim4& other)
{
    dims[0] = other.dims[0];
    dims[1] = other.dims[1];
    dims[2] = other.dims[2];
    dims[3] = other.dims[3];
}

dim4::dim4(const unsigned ndims_, const dim_t * const dims_)
{
    for (unsigned i = 0; i < 4; i++) {
        dims[i] = ndims_ > i ? dims_[i] : 1;
    }
}


dim_t
dim4::elements() const
{
    return dims[0] * dims[1] * dims[2] * dims[3];
}

dim_t
dim4::elements()
{
    return static_cast<const dim4&>(*this).elements();
}

dim_t
dim4::ndims() const
{
    int num = elements();
    if (num == 0) return 0;
    if (num == 1) return 1;

    if (dims[3] != 1) return 4;
    if (dims[2] != 1) return 3;
    if (dims[1] != 1) return 2;

    return 1;
}

dim_t
dim4::ndims()
{
    return static_cast<const dim4&>(*this).ndims();
}

const dim_t&
dim4::operator[](const unsigned dim) const
{
    return dims[dim];
}

dim_t &
dim4::operator[](const unsigned dim)
{
    return const_cast<dim_t&>(static_cast<const dim4&>((*this))[dim]);
}

bool
dim4::operator==(const dim4 &other) const
{
    bool ret = true;
    for(unsigned i = 0; i < 4 && ret; i++) {
        ret = (*this)[i] == other[i];
    }
    return ret;
}

bool
dim4::operator!=(const dim4 &other) const
{
    return !((*this) == other);
}

dim4&
dim4::operator*=(const dim4 &other)
{
    for(unsigned i = 0; i < 4; i++) {
        (*this)[i] *= other[i];
    }
    return *this;
}

dim4&
dim4::operator+=(const dim4 &other)
{
    for(unsigned i = 0; i < 4; i++) {
        (*this)[i] = (*this)[i] + other[i];
    }
    return *this;
}

dim4&
dim4::operator-=(const dim4 &other)
{
    for(unsigned i = 0; i < 4; i++) {
        (*this)[i] = (*this)[i] - other[i];
    }
    return *this;
}

dim4 operator+(const dim4& first, const dim4& second)
{
    dim4 dims;
    for(unsigned i = 0; i < 4; i++) {
        dims[i] = first[i] + second[i];
    }
    return dims;
}

dim4 operator-(const dim4& first, const dim4& second)
{
    dim4 dims;
    for(unsigned i = 0; i < 4; i++) {
        dims[i] = first[i] - second[i];
    }
    return dims;
}

dim4 operator*(const dim4& first, const dim4& second)
{
    dim4 dims;
    for(unsigned i = 0; i < 4; i++) {
        dims[i] = first[i] * second[i];
    }
    return dims;
}


bool
isEnd(const af_seq &seq)    { return (seq.end <= -1); }

bool
isSpan(const af_seq &seq)   { return (seq.step == 0 && seq.begin == 1 && seq.end == 1); }

size_t
seqElements(const af_seq &seq) {
    size_t out = 0;
    if      (seq.step > DBL_MIN)    { out = ((seq.end - seq.begin) / abs(seq.step)) + 1;    }
    else if (seq.step < -DBL_MIN)   { out = ((seq.begin - seq.end) / abs(seq.step)) + 1;    }
    else                            { out = numeric_limits<size_t>::max();                  }

    return out;
}

dim_t calcDim(const af_seq &seq, const dim_t &parentDim)
{
    dim_t outDim = 1;
    if  (isSpan(seq)) {
        outDim = parentDim;
    } else if (isEnd(seq)) {
        if(seq.begin == -1) {   // only end is passed as seq
            outDim = 1;
        } else if (seq.begin < 0) {
            af_seq temp = {parentDim + seq.begin,
                           parentDim + seq.end,
                           seq.step};
            outDim = seqElements(temp);
        } else {    // end is passed as a part of seq
            af_seq temp = {seq.begin, parentDim + seq.end, seq.step};
            outDim = seqElements(temp);
        }
    } else {
        DIM_ASSERT(1, seq.begin >= -DBL_MIN && seq.begin < parentDim);
        DIM_ASSERT(1, seq.end < parentDim);
        outDim = seqElements(seq);
    }

    return outDim;
}
}

using af::dim4;
using std::vector;

dim4
toDims(const vector<af_seq>& seqs, const dim4 &parentDims)
{
    dim4 outDims(1, 1, 1, 1);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        outDims[i] = af::calcDim(seqs[i], parentDims[i]);
        if (outDims[i] > parentDims[i])
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
    }
    return outDims;
}

dim4
toOffset(const vector<af_seq>& seqs, const dim4 &parentDims)
{
    dim4 outOffsets(0, 0, 0, 0);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if (seqs[i].step !=0 && seqs[i].begin >= 0) {
            outOffsets[i] = seqs[i].begin;
        } else if (seqs[i].begin <= -1) {
            outOffsets[i] = parentDims[i] + seqs[i].begin;
        } else {
            outOffsets[i] = 0;
        }

        if (outOffsets[i] >= parentDims[i])
            AF_ERROR("Index out of range", AF_ERR_SIZE);
    }
    return outOffsets;
}

dim4
toStride(const vector<af_seq>& seqs, const af::dim4 &parentDims)
{
    dim4 out(calcStrides(parentDims));
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if  (seqs[i].step != 0) {   out[i] *= seqs[i].step; }
    }
    return out;
}
