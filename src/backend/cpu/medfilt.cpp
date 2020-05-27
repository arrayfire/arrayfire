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

template<typename T>
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const af::borderType pad) {
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(kernel::medfilt1<T>, out, in, w_wid, pad);
    return out;
}

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const af::borderType pad) {
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(kernel::medfilt2<T>, out, in, w_len, w_wid, pad);
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> medfilt1<T>(const Array<T> &in, const int w_wid, \
                                  const af::borderType);               \
    template Array<T> medfilt2<T>(const Array<T> &in, const int w_len, \
                                  const int w_wid, const af::borderType);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
