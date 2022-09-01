/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <convolve.hpp>
#include <err_oneapi.hpp>
#include <iir.hpp>
//#include <kernel/iir.hpp>
#include <math.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace oneapi {
template<typename T>
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x) {
    ONEAPI_NOT_SUPPORTED("");
    Array<T> y = createEmptyArray<T>(dim4(1));
    return y;
}

#define INSTANTIATE(T)                                          \
    template Array<T> iir(const Array<T> &b, const Array<T> &a, \
                          const Array<T> &x);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
}  // namespace oneapi
