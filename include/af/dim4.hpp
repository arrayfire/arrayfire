/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus

#include <ostream>
#include <istream>
#include <vector>
#if __cplusplus > 199711L // Necessary for NVCC
//#include <initializer_list>
#endif
#include <af/defines.h>
#include <af/seq.h>


namespace af
{
class AFAPI dim4
{
    public:
    dim_t dims[4]; //FIXME: Make this C compatiable
    dim4(); //deleted
public:
#if __cplusplus > 199711L
    //dim4(std::initializer_list<dim_t> dim_vals);
#endif
    dim4(   dim_t first,
            dim_t second = 1,
            dim_t third = 1,
            dim_t fourth = 1);
    dim4(const dim4& other);
    dim4(const unsigned ndims, const dim_t * const dims);
    dim_t elements();
    dim_t elements() const;
    dim_t ndims();
    dim_t ndims() const;
    bool operator==(const dim4& other) const;
    bool operator!=(const dim4& other) const;
    dim4& operator*=(const dim4& other);
    dim4& operator+=(const dim4& other);
    dim4& operator-=(const dim4& other);
    dim_t& operator[](const unsigned dim);
    const dim_t& operator[](const unsigned dim) const;
            dim_t* get()         { return dims; }
    const   dim_t* get() const   { return dims; }
};

dim4 operator+(const dim4& first, const dim4& second);
dim4 operator-(const dim4& first, const dim4& second);
dim4 operator*(const dim4& first, const dim4& second);

static inline
std::ostream&
operator<<(std::ostream& ostr, const dim4& dims)
{
    ostr << dims[0] << " "
         << dims[1] << " "
         << dims[2] << " "
         << dims[3];
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

AFAPI bool isSpan(const af_seq &seq);

AFAPI size_t seqElements(const af_seq &seq);

AFAPI dim_t calcDim(const af_seq &seq, const dim_t &parentDim);
}

#endif
