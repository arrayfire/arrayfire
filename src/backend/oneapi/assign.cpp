/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <assign.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using af::dim4;
using common::half;

namespace oneapi {

template<typename T>
void assign(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs) {
    ONEAPI_NOT_SUPPORTED("");
    return;
}

#define INSTANTIATE(T)                                                \
    template void assign<T>(Array<T> & out, const af_index_t idxrs[], \
                            const Array<T>& rhs);

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
