/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include <af/seq.h>

#include "error.hpp"

#include <cmath>

namespace af {
int end = -1;
seq span(af_span);

void seq::init(double begin, double end, double step) {
    this->s.begin = begin;
    this->s.end   = end;
    this->s.step  = step;
    if (step != 0) {  // Not Span
        size = std::fabs((end - begin) / step) + 1;
    } else {
        size = 0;
    }
}

#ifndef signbit
// wtf windows?!
inline int signbit(double x) {
    if (x < 0) { return -1; }
    return 0;
}
#endif

seq::~seq() = default;

seq::seq(double length) : s{}, size{}, m_gfor(false) {
    if (length < 0) {
        init(0, length, 1);
    } else {
        init(0, length - 1, 1);
    }
}

seq::seq(const af_seq& s_) : s{}, size{}, m_gfor(false) {
    init(s_.begin, s_.end, s_.step);
}

seq& seq::operator=(const af_seq& s_) {
    init(s_.begin, s_.end, s_.step);
    return *this;
}

seq::seq(double begin, double end, double step) : s{}, size{}, m_gfor(false) {
    if (step == 0) {
        if (begin != end) {  // Span
            AF_THROW_ERR("Invalid step size", AF_ERR_ARG);
        }
    }
    if ((signbit(end) == signbit(begin)) &&
        (signbit(end - begin) != signbit(step))) {
        AF_THROW_ERR("Sequence is invalid", AF_ERR_ARG);
    }
    init(begin, end, step);
}

seq::seq(seq other,  // NOLINT(performance-unnecessary-value-param)
         bool is_gfor)
    : s(other.s), size(other.size), m_gfor(is_gfor) {}

seq::operator array() const {
    double diff = s.end - s.begin;
    dim_t len   = static_cast<int>(
        (diff + std::fabs(s.step) * (signbit(diff) == 0 ? 1 : -1)) / s.step);

    array tmp = (m_gfor) ? range(1, 1, 1, len, 3) : range(len);

    array res = s.begin + s.step * tmp;
    return res;
}
}  // namespace af
