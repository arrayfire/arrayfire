/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <kernel/range.hpp>
#include <range.hpp>

#include <Array.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> range(const dim4& dims, const int seq_dim) {
    // Set dimension along which the sequence should be
    // Other dimensions are simply tiled
    int _seq_dim = seq_dim;
    if (seq_dim < 0) {
        _seq_dim = 0;  // column wise sequence
    }

    Array<T> out = createEmptyArray<T>(dims);
    switch (_seq_dim) {
        case 0: getQueue().enqueue(kernel::range<T, 0>, out); break;
        case 1: getQueue().enqueue(kernel::range<T, 1>, out); break;
        case 2: getQueue().enqueue(kernel::range<T, 2>, out); break;
        case 3: getQueue().enqueue(kernel::range<T, 3>, out); break;
        default: AF_ERROR("Invalid rep selection", AF_ERR_ARG);
    }

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> range<T>(const af::dim4& dims, const int seq_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace arrayfire
