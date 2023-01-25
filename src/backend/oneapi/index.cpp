/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <index.hpp>

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[]) {
    ONEAPI_NOT_SUPPORTED("");
    Array<T> out = createEmptyArray<T>(af::dim4(1));
    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const af_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace oneapi
}  // namespace arrayfire
