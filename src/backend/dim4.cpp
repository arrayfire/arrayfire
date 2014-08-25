#include <limits>
#include <numeric>
#include <vector>
#include <iostream>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>

namespace af
{

using std::vector;
using std::numeric_limits;

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
    for(int i = 3; i >= 0; i--) {
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
isSpan(const af_seq &seq)          { return (seq.step == 0); }

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
    dim4 out(1, 1, 1, 1);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if  (isSpan(seqs[i]))   { out[i] = parentDims[i];          }
        else                    { out[i] = seqElements(seqs[i]);   }
    }
    return out;
}

dim4
toOffset(const vector<af_seq>& seqs)
{
    dim4 out(0, 0, 0, 0);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if      (seqs[i].step != 0) {   out[i] = seqs[i].begin; }
        else                        {   out[i] = 0;             }
    }
    return out;
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
