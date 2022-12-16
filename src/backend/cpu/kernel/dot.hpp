/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <complex>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
T conj(T x) {
    return x;
}

template<>
cfloat conj<cfloat>(cfloat c) {
    return std::conj(c);
}
template<>
cdouble conj<cdouble>(cdouble c) {
    return std::conj(c);
}

template<typename T, bool conjugate, bool both_conjugate>
void dot(Param<T> output, CParam<T> lhs, CParam<T> rhs, af_mat_prop optLhs,
         af_mat_prop optRhs) {
    UNUSED(optLhs);
    UNUSED(optRhs);
    int N = lhs.dims(0);

    T out       = 0;
    const T *pL = lhs.get();
    const T *pR = rhs.get();

    for (int i = 0; i < N; i++)
        out += (conjugate ? kernel::conj(pL[i]) : pL[i]) * pR[i];

    if (both_conjugate) out = kernel::conj(out);

    *output.get() = out;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
