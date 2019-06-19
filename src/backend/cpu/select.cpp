/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/select.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <select.hpp>

using af::dim4;

namespace cpu {

template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    getQueue().enqueue(kernel::select<T>, out, cond, a, b);
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const double &b) {
    getQueue().enqueue(kernel::select_scalar<T, flip>, out, cond, a, b);
}

#define INSTANTIATE(T)                                                        \
    template void select<T>(Array<T> & out, const Array<char> &cond,          \
                            const Array<T> &a, const Array<T> &b);            \
    template void select_scalar<T, true>(Array<T> & out,                      \
                                         const Array<char> &cond,             \
                                         const Array<T> &a, const double &b); \
    template void select_scalar<T, false>(Array<T> & out,                     \
                                          const Array<char> &cond,            \
                                          const Array<T> &a, const double &b);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
