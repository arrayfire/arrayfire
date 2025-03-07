/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/constants.h>
#include <algorithm>
#include <cmath>
#include "backend.hpp"

namespace arrayfire {
namespace cpu {
static inline dim_t trimIndex(int const& idx, dim_t const& len) {
    int ret_val = idx;
    if (ret_val < 0) {
        int offset = (abs(ret_val) - 1) % len;
        ret_val    = offset;
    } else if (ret_val >= (int)len) {
        int offset = abs(ret_val) % len;
        ret_val    = len - offset - 1;
    }
    return ret_val;
}

static inline unsigned getIdx(af::dim4 const& strides, int i, int j = 0,
                              int k = 0, int l = 0) {
    return (l * strides[3] + k * strides[2] + j * strides[1] + i * strides[0]);
}

template<typename T>
void gaussian1D(T* out, int const dim, double sigma = 0.0) {
    if (!(sigma > 0)) sigma = 0.25 * dim;

    T sum = (T)0;
    for (int i = 0; i < dim; i++) {
        int x = i - (dim - 1) / 2;
        T el  = 1. / std::sqrt(2 * af::Pi * sigma * sigma) *
               std::exp(-((x * x) / (2 * (sigma * sigma))));
        out[i] = el;
        sum += el;
    }

    for (int k = 0; k < dim; k++) out[k] /= sum;
}
}  // namespace cpu
}  // namespace arrayfire
