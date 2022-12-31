/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <convolve.hpp>

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace oneapi {

template<typename T, typename accT>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter,
                   Array<accT> const& r_filter, const bool expand) {
    ONEAPI_NOT_SUPPORTED("");
    Array<T> out = createEmptyArray<T>(dim4(1));
    return out;
}

#define INSTANTIATE(T, accT)                                                  \
    template Array<T> convolve2<T, accT>(Array<T> const&, Array<accT> const&, \
                                         Array<accT> const&, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)
INSTANTIATE(intl, float)
INSTANTIATE(uintl, float)

}  // namespace oneapi
}  // namespace arrayfire
