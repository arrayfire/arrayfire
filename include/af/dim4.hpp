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
#include <af/defines.h>
#include <af/seq.h>


namespace af
{
class AFAPI dim4
{
    public:
    dim_t dims[4]; //FIXME: Make this C compatible
    dim4(); //deleted
public:
    dim4(   dim_t first,
            dim_t second = 1,
            dim_t third = 1,
            dim_t fourth = 1);
    dim4(const dim4& other);
    dim4(const unsigned ndims, const dim_t * const dims);
    inline dim_t elements() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
    inline dim_t ndims() const {
        if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0) {
            return 0;
        }
        if (dims[3] != 1) return 4;
        if (dims[2] != 1) return 3;
        if (dims[1] != 1) return 2;

        return 1;
    }
    bool operator==(const dim4& other) const;
    bool operator!=(const dim4& other) const;
    dim4& operator*=(const dim4& other);
    dim4& operator+=(const dim4& other);
    dim4& operator-=(const dim4& other);
    inline dim_t& operator[](const unsigned dim) { return dims[dim]; }
    inline const dim_t& operator[](const unsigned dim) const { return dims[dim]; }
    inline dim_t* get() { return dims; }
    inline const dim_t* get() const { return dims; }
};

AFAPI dim4 operator+(const dim4& first, const dim4& second);
AFAPI dim4 operator-(const dim4& first, const dim4& second);
AFAPI dim4 operator*(const dim4& first, const dim4& second);

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
