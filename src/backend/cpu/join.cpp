/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <join.hpp>
#include <kernel/join.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <algorithm>

using common::half;

namespace cpu {

template<typename T>
Array<T> join(const int dim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    af::dim4 odims;
    af::dim4 fdims = first.dims();
    af::dim4 sdims = second.dims();

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = fdims[i] + sdims[i];
        } else {
            odims[i] = fdims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(odims);
    std::vector<CParam<T>> v{first, second};
    getQueue().enqueue(kernel::join<T>, dim, out, v, 2);

    return out;
}

template<typename T>
Array<T> join(const int dim, const std::vector<Array<T>> &inputs) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    af::dim4 odims;
    const dim_t n_arrays = inputs.size();
    std::vector<af::dim4> idims(n_arrays);

    dim_t dim_size = 0;
    for (unsigned i = 0; i < idims.size(); i++) {
        idims[i] = inputs[i].dims();
        dim_size += idims[i][dim];
    }

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = dim_size;
        } else {
            odims[i] = idims[0][i];
        }
    }

    std::vector<Array<T> *> input_ptrs(inputs.size());
    std::transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);
    std::vector<CParam<T>> inputParams(inputs.begin(), inputs.end());
    Array<T> out = createEmptyArray<T>(odims);

    getQueue().enqueue(kernel::join<T>, dim, out, inputParams, n_arrays);

    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> join<T>(const int dim, const Array<T> &first, \
                              const Array<T> &second);

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
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T)                       \
    template Array<T> join<T>(const int dim, \
                              const std::vector<Array<T>> &inputs);

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
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace cpu
