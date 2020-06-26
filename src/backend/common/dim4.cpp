/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <af/dim4.hpp>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>

namespace af {

#if AF_COMPILER_CXX_STATIC_ASSERT
static_assert(std::is_standard_layout<dim4>::value,
              "af::dim4 must be a standard layout type");
#endif

using std::abs;
using std::numeric_limits;

dim4::dim4() : dims{0, 0, 0, 0} {}

dim4::dim4(dim_t first, dim_t second, dim_t third, dim_t fourth)
    : dims{first, second, third, fourth} {}

dim4::dim4(const dim4& other)
    : dims{other.dims[0], other.dims[1], other.dims[2], other.dims[3]} {}

dim4::dim4(const unsigned ndims_, const dim_t* const dims_) : dims{} {
    for (unsigned i = 0; i < 4; i++) { dims[i] = ndims_ > i ? dims_[i] : 1; }
}

dim4& dim4::operator=(dim4 other) noexcept {
    std::swap(dims, other.dims);
    return *this;
}

dim_t dim4::elements() const { return dims[0] * dims[1] * dims[2] * dims[3]; }

dim_t dim4::elements() { return static_cast<const dim4&>(*this).elements(); }

dim_t dim4::ndims() const {
    dim_t num = elements();
    if (num == 0) { return 0; }
    if (num == 1) { return 1; }

    if (dims[3] != 1) { return 4; }
    if (dims[2] != 1) { return 3; }
    if (dims[1] != 1) { return 2; }

    return 1;
}

dim_t dim4::ndims() { return static_cast<const dim4&>(*this).ndims(); }

const dim_t& dim4::operator[](const unsigned dim) const { return dims[dim]; }

dim_t& dim4::operator[](const unsigned dim) {
    return const_cast<dim_t&>(static_cast<const dim4&>((*this))[dim]);
}

bool dim4::operator==(const dim4& other) const {
    bool ret = true;
    for (unsigned i = 0; i < 4 && ret; i++) { ret = (*this)[i] == other[i]; }
    return ret;
}

bool dim4::operator!=(const dim4& other) const { return !((*this) == other); }

dim4& dim4::operator*=(const dim4& other) {
    for (unsigned i = 0; i < 4; i++) { (*this)[i] *= other[i]; }
    return *this;
}

dim4& dim4::operator+=(const dim4& other) {
    for (unsigned i = 0; i < 4; i++) { (*this)[i] = (*this)[i] + other[i]; }
    return *this;
}

dim4& dim4::operator-=(const dim4& other) {
    for (unsigned i = 0; i < 4; i++) { (*this)[i] = (*this)[i] - other[i]; }
    return *this;
}

dim4 operator+(const dim4& first, const dim4& second) {
    dim4 dims;
    for (unsigned i = 0; i < 4; i++) { dims[i] = first[i] + second[i]; }
    return dims;
}

dim4 operator-(const dim4& first, const dim4& second) {
    dim4 dims;
    for (unsigned i = 0; i < 4; i++) { dims[i] = first[i] - second[i]; }
    return dims;
}

dim4 operator*(const dim4& first, const dim4& second) {
    dim4 dims;
    for (unsigned i = 0; i < 4; i++) { dims[i] = first[i] * second[i]; }
    return dims;
}

bool hasEnd(const af_seq& seq) { return (seq.begin <= -1 || seq.end <= -1); }

bool isSpan(const af_seq& seq) {
    return (seq.step == 0 && seq.begin == 1 && seq.end == 1);
}

size_t seqElements(const af_seq& seq) {
    size_t out = 0;
    if (seq.step > DBL_MIN) {
        out = ((seq.end - seq.begin) / abs(seq.step)) + 1;
    } else if (seq.step < -DBL_MIN) {
        out = ((seq.begin - seq.end) / abs(seq.step)) + 1;
    } else {
        out = numeric_limits<size_t>::max();
    }

    return out;
}

dim_t calcDim(const af_seq& seq, const dim_t& parentDim) {
    dim_t outDim = 1;
    if (isSpan(seq)) {
        outDim = parentDim;
    } else if (hasEnd(seq)) {
        af_seq temp = {seq.begin, seq.end, seq.step};
        if (seq.begin < 0) { temp.begin += parentDim; }
        if (seq.end < 0) { temp.end += parentDim; }
        outDim = seqElements(temp);
    } else {
        DIM_ASSERT(1, seq.begin >= -DBL_MIN && seq.begin < parentDim);
        DIM_ASSERT(1, seq.end < parentDim);
        outDim = seqElements(seq);
    }

    return outDim;
}

}  // namespace af
