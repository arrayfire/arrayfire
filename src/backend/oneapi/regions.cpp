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
// #include <kernel/regions.hpp>
#include <regions.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> regions(const Array<char> &in, af_connectivity connectivity) {
    ONEAPI_NOT_SUPPORTED("regions Not supported");

    const af::dim4 &dims = in.dims();
    Array<T> out         = createEmptyArray<T>(dims);
    // kernel::regions<T>(out, in, connectivity == AF_CONNECTIVITY_8, 2);
    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 af_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace oneapi
}  // namespace arrayfire
