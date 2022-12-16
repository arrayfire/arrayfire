/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <medfilt.hpp>

#include <Array.hpp>
#include <kernel/medfilt.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <functional>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename T>
using medianFilter1 = std::function<void(Param<T>, CParam<T>, dim_t)>;

template<typename T>
using medianFilter2 = std::function<void(Param<T>, CParam<T>, dim_t, dim_t)>;

template<typename T>
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const af::borderType pad) {
    static const medianFilter1<T> funcs[2] = {
        kernel::medfilt1<T, AF_PAD_ZERO>,
        kernel::medfilt1<T, AF_PAD_SYM>,
    };
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(funcs[static_cast<int>(pad)], out, in, w_wid);
    return out;
}

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const af::borderType pad) {
    static const medianFilter2<T> funcs[2] = {
        kernel::medfilt2<T, AF_PAD_ZERO>,
        kernel::medfilt2<T, AF_PAD_SYM>,
    };
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(funcs[static_cast<int>(pad)], out, in, w_len, w_wid);
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
}  // namespace arrayfire
