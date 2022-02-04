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
#include <err_cuda.hpp>
#include <join.hpp>
#include <kernel/join.hpp>

#include <algorithm>
#include <stdexcept>

using common::half;

namespace cuda {

af::dim4 calcOffset(const af::dim4 &dims, const int dim) {
    af::dim4 offset;
    offset[0] = (dim == 0) * dims[0];
    offset[1] = (dim == 1) * dims[1];
    offset[2] = (dim == 2) * dims[2];
    offset[3] = (dim == 3) * dims[3];
    return offset;
}

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

    af::dim4 zero(0, 0, 0, 0);

    kernel::join<T>(out, first, zero, dim);
    kernel::join<T>(out, second, calcOffset(fdims, dim), dim);

    return out;
}

template<typename T>
void join_wrapper(const int dim, Array<T> &out,
                  const std::vector<Array<T>> &inputs) {
    af::dim4 zero(0, 0, 0, 0);
    af::dim4 d = zero;

    kernel::join<T>(out, inputs[0], zero, dim);
    for (size_t i = 1; i < inputs.size(); i++) {
        d += inputs[i - 1].dims();
        kernel::join<T>(out, inputs[i], calcOffset(d, dim), dim);
    }
}

template<typename T>
void join(Array<T> &out, const int dim, const std::vector<Array<T>> &inputs) {
    std::vector<Array<T> *> input_ptrs(inputs.size());
    std::transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);

    join_wrapper<T>(dim, out, inputs);
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
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T)                                   \
    template void join<T>(Array<T> & out, const int dim, \
                          const std::vector<Array<T>> &inputs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace cuda
