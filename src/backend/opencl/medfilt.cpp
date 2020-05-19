/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/medfilt.hpp>
#include <medfilt.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace opencl {

template<typename T, af_border_type pad>
Array<T> medfilt1(const Array<T> &in, dim_t w_wid) {
    ARG_ASSERT(2, (w_wid <= kernel::MAX_MEDFILTER1_LEN));
    ARG_ASSERT(2, (w_wid % 2 != 0));

    const dim4 &dims = in.dims();

    Array<T> out = createEmptyArray<T>(dims);

    kernel::medfilt1<T>(out, in, w_wid, pad);

    return out;
}

template<typename T, af_border_type pad>
Array<T> medfilt2(const Array<T> &in, dim_t w_len, dim_t w_wid) {
    UNUSED(w_wid);
    ARG_ASSERT(2, (w_len == w_wid));
    ARG_ASSERT(2, (w_len % 2 != 0));
    ARG_ASSERT(2, (w_len <= kernel::MAX_MEDFILTER2_LEN));

    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::medfilt2<T>(out, in, pad, w_len, w_wid);
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
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace opencl
