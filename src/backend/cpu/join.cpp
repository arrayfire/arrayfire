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
#include <cassert>
#include <numeric>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    // Compute output dims
    const dim4 &fdims = first.dims();
    const dim4 &sdims = second.dims();
    // All dimensions except join dimension must be equal
    assert((jdim == 0 ? true : fdims.dims[0] == sdims.dims[0]) &&
           (jdim == 1 ? true : fdims.dims[1] == sdims.dims[1]) &&
           (jdim == 2 ? true : fdims.dims[2] == sdims.dims[2]) &&
           (jdim == 3 ? true : fdims.dims[3] == sdims.dims[3]));

    // compute output dms
    dim4 odims(fdims);
    odims.dims[jdim] += sdims.dims[jdim];
    Array<T> out = createEmptyArray<T>(odims);
    std::vector<CParam<T>> v{first, second};
    getQueue().enqueue(kernel::join<T>, jdim, out, v, 2);

    return out;
}

template<typename T>
void join(Array<T> &out, const int jdim, const std::vector<Array<T>> &inputs) {
    const dim_t n_arrays = inputs.size();
    if (n_arrays == 0) return;

    // avoid buffer overflow
    const dim4 &odims{out.dims()};
    const dim4 &fdims{inputs[0].dims()};
    // All dimensions of inputs needs to be equal except for the join
    // dimension
    assert(std::all_of(inputs.begin(), inputs.end(),
                       [jdim, &fdims](const Array<T> &in) {
                           bool eq{true};
                           for (int i = 0; i < 4; ++i) {
                               if (i != jdim) {
                                   eq &= fdims.dims[i] == in.dims().dims[i];
                               };
                           };
                           return eq;
                       }));
    // All dimensions of out needs to cover all input dimensions
    assert(
        (odims.dims[0] >= fdims.dims[0]) && (odims.dims[1] >= fdims.dims[1]) &&
        (odims.dims[2] >= fdims.dims[2]) && (odims.dims[3] >= fdims.dims[3]));
    // The join dimension of out needs to be larger than the
    // sum of all input join dimensions
    assert(odims.dims[jdim] >=
           std::accumulate(inputs.begin(), inputs.end(), 0,
                           [jdim](dim_t dim, const Array<T> &in) {
                               return dim += in.dims()[jdim];
                           }));
    assert(out.strides().dims[0] == 1);

    std::vector<Array<T> *> input_ptrs(n_arrays);
    std::transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);
    std::vector<CParam<T>> inputParams(inputs.begin(), inputs.end());

    getQueue().enqueue(kernel::join<T>, jdim, out, inputParams, n_arrays);
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
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace cpu
}  // namespace arrayfire
