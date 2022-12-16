/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <iota.hpp>
#include <kernel/iota.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

using arrayfire::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> iota(const dim4 &dims, const dim4 &tile_dims) {
    dim4 outdims = dims * tile_dims;

    Array<T> out = createEmptyArray<T>(outdims);

    getQueue().enqueue(kernel::iota<T>, out, dims);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> iota<T>(const af::dim4 &dims, const af::dim4 &tile_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace arrayfire
