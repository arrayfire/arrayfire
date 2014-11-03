/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <ostream>
#include <istream>
#include <vector>
#if __cplusplus > 199711L // Necessary for NVCC
//#include <initializer_list>
#endif
#include <af/defines.h>
#include <af/array.h>


namespace af
{
class AFAPI dim4
{
    public:
    dim_type dims[4]; //FIXME: Make this C compatiable
    dim4(); //deleted
public:
#if __cplusplus > 199711L
    //dim4(std::initializer_list<dim_type> dim_vals);
#endif
    dim4(   dim_type first,
            dim_type second = 1,
            dim_type third = 1,
            dim_type fourth = 1);
    dim4(const dim4& other);
    dim_type elements();
    dim_type elements() const;
    dim_type ndims();
    dim_type ndims() const;
    bool operator==(const dim4& other) const;
    bool operator!=(const dim4& other) const;
    dim4& operator*=(const dim4& other);
    dim4& operator+=(const dim4& other);
    dim4& operator-=(const dim4& other);
    dim_type& operator[](const unsigned dim);
    const dim_type& operator[](const unsigned dim) const;
            dim_type* get()         { return dims; }
    const   dim_type* get() const   { return dims; }
};

static inline
std::ostream&
operator<<(std::ostream& ostr, const dim4& dims)
{
    ostr << dims[0] << " "
         << dims[1] << " "
         << dims[2] << " "
         << dims[3] << "\n";
    return ostr;
}

static inline
std::istream&
operator>>(std::istream& istr, dim4& dims)
{
    istr >> dims[0]
         >> dims[1]
         >> dims[2]
         >> dims[3];
    return istr;
}

bool isSpan(const af_seq &seq);

size_t seqElements(const af_seq &seq);

dim4 toDims(const std::vector<af_seq>& seqs, af::dim4 parentDims);

dim4 toOffset(const std::vector<af_seq>& seqs);

dim4 toStride(const std::vector<af_seq>& seqs, af::dim4 parentDims);
}
