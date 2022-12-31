/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <join.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using af::dim4;
using arrayfire::common::half;
using std::transform;
using std::vector;

namespace arrayfire {
namespace oneapi {
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
    ONEAPI_NOT_SUPPORTED("");
    Array<T> out = createEmptyArray<T>(af::dim4(1));
    return out;
}

template<typename T>
void join_wrapper(const int dim, Array<T> &out,
                  const vector<Array<T>> &inputs) {
    ONEAPI_NOT_SUPPORTED("");
}

template<typename T>
void join(Array<T> &out, const int dim, const vector<Array<T>> &inputs) {
    ONEAPI_NOT_SUPPORTED("");
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
}  // namespace oneapi
}  // namespace arrayfire
