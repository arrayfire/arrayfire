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
#include <kernel/lookup.hpp>
#include <lookup.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>

#include <cassert>

using arrayfire::common::half;

namespace arrayfire {
namespace cpu {
template<typename in_t, typename idx_t>
Array<in_t> lookup(const Array<in_t> &input, const Array<idx_t> &indices,
                   const unsigned dim) {
    assert((af_dtype)dtype_traits<idx_t>::af_type != c32 &&
           (af_dtype)dtype_traits<idx_t>::af_type != c64 &&
           (af_dtype)dtype_traits<idx_t>::af_type != b8);
    assert(dim <= 3);

    dim4 oDims(input.dims());
    oDims[dim] = indices.elements();

    Array<in_t> out = createEmptyArray<in_t>(oDims);
    getQueue().enqueue(kernel::lookup<in_t, idx_t>, out, input, indices, dim);

    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> lookup<T, float>(const Array<T> &, const Array<float> &, \
                                       const unsigned);                        \
    template Array<T> lookup<T, double>(                                       \
        const Array<T> &, const Array<double> &, const unsigned);              \
    template Array<T> lookup<T, int>(const Array<T> &, const Array<int> &,     \
                                     const unsigned);                          \
    template Array<T> lookup<T, unsigned>(                                     \
        const Array<T> &, const Array<unsigned> &, const unsigned);            \
    template Array<T> lookup<T, short>(const Array<T> &, const Array<short> &, \
                                       const unsigned);                        \
    template Array<T> lookup<T, ushort>(                                       \
        const Array<T> &, const Array<ushort> &, const unsigned);              \
    template Array<T> lookup<T, intl>(const Array<T> &, const Array<intl> &,   \
                                      const unsigned);                         \
    template Array<T> lookup<T, uintl>(const Array<T> &, const Array<uintl> &, \
                                       const unsigned);                        \
    template Array<T> lookup<T, uchar>(const Array<T> &, const Array<uchar> &, \
                                       const unsigned);                        \
    template Array<T> lookup<T, half>(const Array<T> &, const Array<half> &,   \
                                      const unsigned);

INSTANTIATE(float);
INSTANTIATE(cfloat);
INSTANTIATE(double);
INSTANTIATE(cdouble);
INSTANTIATE(int);
INSTANTIATE(unsigned);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(uchar);
INSTANTIATE(char);
INSTANTIATE(ushort);
INSTANTIATE(short);
INSTANTIATE(half);
}  // namespace cpu
}  // namespace arrayfire
