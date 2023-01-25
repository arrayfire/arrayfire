/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <kernel/resize.hpp>
#include <resize.hpp>
#include <af/dim4.hpp>
#include <stdexcept>

namespace arrayfire {
namespace oneapi {
template<typename T>
Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                const af_interp_type method) {
    const af::dim4 &iDims = in.dims();
    af::dim4 oDims(odim0, odim1, iDims[2], iDims[3]);
    Array<T> out = createEmptyArray<T>(oDims);

    if constexpr (!(std::is_same_v<T, double> || std::is_same_v<T, cdouble>)) {
        kernel::resize<T>(out, in, method);
    }
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> resize<T>(const Array<T> &in, const dim_t odim0, \
                                const dim_t odim1,                     \
                                const af_interp_type method);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
}  // namespace oneapi
}  // namespace arrayfire
