/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/medfilt.hpp>
#include <medfilt.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace cpu {

template<typename T, af_border_type pad>
Array<T> medfilt1(const Array<T> &in, dim_t w_wid) {
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::medfilt1<T, pad>, out, in, w_wid);

    return out;
}

template<typename T, af_border_type pad>
Array<T> medfilt2(const Array<T> &in, dim_t w_len, dim_t w_wid) {
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::medfilt2<T, pad>, out, in, w_len, w_wid);

    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> medfilt1<T, AF_PAD_ZERO>(const Array<T> &in,             \
                                               dim_t w_wid);                   \
    template Array<T> medfilt1<T, AF_PAD_SYM>(const Array<T> &in,              \
                                              dim_t w_wid);                    \
    template Array<T> medfilt2<T, AF_PAD_ZERO>(const Array<T> &in,             \
                                               dim_t w_len, dim_t w_wid);      \
    template Array<T> medfilt2<T, AF_PAD_SYM>(const Array<T> &in, dim_t w_len, \
                                              dim_t w_wid);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
