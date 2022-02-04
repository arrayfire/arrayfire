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
#include <err_opencl.hpp>
#include <join.hpp>
#include <kernel/join.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using af::dim4;
using common::half;
using std::transform;
using std::vector;

namespace opencl {
dim4 calcOffset(const dim4 &dims, int dim) {
    dim4 offset;
    offset[0] = (dim == 0) ? dims[0] : 0;
    offset[1] = (dim == 1) ? dims[1] : 0;
    offset[2] = (dim == 2) ? dims[2] : 0;
    offset[3] = (dim == 3) ? dims[3] : 0;
    return offset;
}

template<typename T>
Array<T> join(const int dim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    dim4 odims;
    dim4 fdims = first.dims();
    dim4 sdims = second.dims();

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = fdims[i] + sdims[i];
        } else {
            odims[i] = fdims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(odims);

    dim4 zero(0, 0, 0, 0);

    kernel::join<T>(out, first, dim, zero);
    kernel::join<T>(out, second, dim, calcOffset(fdims, dim));

    return out;
}

template<typename T>
void join_wrapper(const int dim, Array<T> &out,
                  const vector<Array<T>> &inputs) {
    dim4 zero(0, 0, 0, 0);
    dim4 d = zero;

    kernel::join<T>(out, inputs[0], dim, zero);
    for (size_t i = 1; i < inputs.size(); i++) {
        d += inputs[i - 1].dims();
        kernel::join<T>(out, inputs[i], dim, calcOffset(d, dim));
    }
}

template<typename T>
void join(Array<T> &out, const int dim, const vector<Array<T>> &inputs) {
    vector<Array<T> *> input_ptrs(inputs.size());
    transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);
    vector<Param> inputParams(inputs.begin(), inputs.end());

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
                          const vector<Array<T>> &inputs);

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
}  // namespace opencl
