/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_oneapi.hpp>
#include <mean.hpp>

#include <common/half.hpp>
// #include <kernel/mean.hpp>
#include <af/dim4.hpp>

using af::dim4;
using common::half;
using std::swap;

namespace oneapi {
template<typename Ti, typename Tw, typename To>
To mean(const Array<Ti>& in) {
    ONEAPI_NOT_SUPPORTED("mean Not supported");

    return To(0);
    // return kernel::meanAll<Ti, Tw, To>(in);
}

template<typename T, typename Tw>
T mean(const Array<T>& in, const Array<Tw>& wts) {
    ONEAPI_NOT_SUPPORTED("mean Not supported");

    return T(0);
    // return kernel::meanAllWeighted<T, Tw>(in, wts);
}

template<typename Ti, typename Tw, typename To>
Array<To> mean(const Array<Ti>& in, const int dim) {
    ONEAPI_NOT_SUPPORTED("mean Not supported");

    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    // kernel::mean<Ti, Tw, To>(out, in, dim);
    return out;
}

template<typename T, typename Tw>
Array<T> mean(const Array<T>& in, const Array<Tw>& wts, const int dim) {
    ONEAPI_NOT_SUPPORTED("mean Not supported");

    dim4 odims   = in.dims();
    odims[dim]   = 1;
    Array<T> out = createEmptyArray<T>(odims);
    // kernel::meanWeighted<T, Tw, T>(out, in, wts, dim);
    return out;
}

#define INSTANTIATE(Ti, Tw, To)                        \
    template To mean<Ti, Tw, To>(const Array<Ti>& in); \
    template Array<To> mean<Ti, Tw, To>(const Array<Ti>& in, const int dim);

INSTANTIATE(double, double, double);
INSTANTIATE(float, float, float);
INSTANTIATE(int, float, float);
INSTANTIATE(unsigned, float, float);
INSTANTIATE(intl, double, double);
INSTANTIATE(uintl, double, double);
INSTANTIATE(short, float, float);
INSTANTIATE(ushort, float, float);
INSTANTIATE(uchar, float, float);
INSTANTIATE(char, float, float);
INSTANTIATE(cfloat, float, cfloat);
INSTANTIATE(cdouble, double, cdouble);
INSTANTIATE(half, float, half);
INSTANTIATE(half, float, float);

#define INSTANTIATE_WGT(T, Tw)                                              \
    template T mean<T, Tw>(const Array<T>& in, const Array<Tw>& wts);       \
    template Array<T> mean<T, Tw>(const Array<T>& in, const Array<Tw>& wts, \
                                  const int dim);

INSTANTIATE_WGT(double, double);
INSTANTIATE_WGT(float, float);
INSTANTIATE_WGT(cfloat, float);
INSTANTIATE_WGT(cdouble, double);
INSTANTIATE_WGT(half, float);

}  // namespace oneapi
