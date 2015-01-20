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
#include <vector>
#include <iostream>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <err_common.hpp>

namespace af
{

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

dim4::dim4( dim_type first,
            dim_type second,
            dim_type third,
            dim_type fourth)
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

dim4::dim4(const unsigned ndims_, const dim_type * const dims_)
{
    for (unsigned i = 0; i < 4; i++) {
        dims[i] = ndims_ > i ? dims_[i] : 1;
    }
}


dim_type
dim4::elements() const
{
    return dims[0] * dims[1] * dims[2] * dims[3];
}

dim_type
dim4::elements()
{
    return static_cast<const dim4&>(*this).elements();
}

dim_type
dim4::ndims() const
{
    dim_type ret = 4;
    for(int i = 3; i >= 1; i--) {
        if(dims[i] == 1)    {   ret--;  }
        else                {   break;  }
    }
    return ret;
}

dim_type
dim4::ndims()
{
    return static_cast<const dim4&>(*this).ndims();
}

const dim_type&
dim4::operator[](const unsigned dim) const
{
    return dims[dim];
}

dim_type &
dim4::operator[](const unsigned dim)
{
    return const_cast<dim_type&>(static_cast<const dim4&>((*this))[dim]);
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

bool
isEnd(const af_seq &seq)    { return (seq.end <= -1); }

bool
isSpan(const af_seq &seq)   { return (seq.step == 0 && seq.begin == 1 && seq.end == 1); }

size_t
seqElements(const af_seq &seq) {
    size_t out = 0;
    if      (seq.step > 0)  { out = ((seq.end - seq.begin) / abs(seq.step)) + 1;    }
    else if (seq.step < 0)  { out = ((seq.begin - seq.end) / abs(seq.step)) + 1;    }
    else                    { out = numeric_limits<size_t>::max();                  }

    return out;
}

dim4
toDims(const vector<af_seq>& seqs, dim4 parentDims)
{
    dim4 outDims(1, 1, 1, 1);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if  (isSpan(seqs[i])) {
            outDims[i] = parentDims[i];
        } else if (isEnd(seqs[i])) {
            if(seqs[i].begin == -1) {   // only end is passed as seq
                outDims[i] = 1;
            } else {    // end is passed as a part of seq
                af_seq temp = {seqs[i].begin, parentDims[i] + seqs[i].end, seqs[i].step};
                outDims[i] = seqElements(temp);
            }
        } else if (seqs[i].begin < 0) {
            // This will throw an error for invalid sequence
            // FIXME
            // Allow reverse sequence, ie. end, 0, -1.
            // Check for seq outDims of bounds on greater side
            AF_ERROR("Sequence out of bounds", AF_ERR_INVALID_ARG);
        } else {
            outDims[i] = seqElements(seqs[i]);
        }

        if (outDims[i] > parentDims[i])
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
    }

    return outDims;
}

dim4
toOffset(const vector<af_seq>& seqs, dim4 parentDims)
{
    dim4 outOffsets(0, 0, 0, 0);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if      (seqs[i].step != 0) {   outOffsets[i] = seqs[i].begin; }
        else if (isEnd(seqs[i]) && seqs[i].begin == -1) { outOffsets[i] = parentDims[i] - 1; }
        else    { outOffsets[i] = 0; }
        if (outOffsets[i] >= parentDims[i])
            AF_ERROR("Index out of range", AF_ERR_SIZE);
    }
    return outOffsets;
}

dim4
toStride(const vector<af_seq>& seqs, af::dim4 parentDims)
{
    dim4 out(calcStrides(parentDims));
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if  (seqs[i].step != 0) {   out[i] *= seqs[i].step; }
    }
    return out;
}
}
