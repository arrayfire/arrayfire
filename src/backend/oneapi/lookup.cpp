/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <lookup.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <kernel/lookup.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace oneapi {
template<typename in_t, typename idx_t>
Array<in_t> lookup(const Array<in_t> &input, const Array<idx_t> &indices,
                   const unsigned dim) {
    const dim4 &iDims = input.dims();

    dim4 oDims(1);
    for (int d = 0; d < 4; ++d) {
        oDims[d] = (d == int(dim) ? indices.elements() : iDims[d]);
    }

    Array<in_t> out = createEmptyArray<in_t>(oDims);

    if constexpr ((!(std::is_same_v<in_t, double> || std::is_same_v<in_t, cdouble>)) &&
                  (!(std::is_same_v<idx_t, double> || std::is_same_v<idx_t, cdouble>))) {
      kernel::lookup<in_t, idx_t>(out, input, indices, dim);
    }

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
                                      const unsigned)

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
}  // namespace oneapi
}  // namespace arrayfire
