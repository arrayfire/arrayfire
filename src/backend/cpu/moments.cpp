/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cpu.hpp>
#include <kernel/moments.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/defines.h>

namespace cpu {

static inline int bitCount(int v) {
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

using af::dim4;

template <typename T>
Array<float> moments(const Array<T> &in, const af_moment_type moment) {
    in.eval();
    dim4 odims, idims = in.dims();
    dim_t moments_dim = bitCount(moment);

    odims[0] = moments_dim;
    odims[1] = 1;
    odims[2] = idims[2];
    odims[3] = idims[3];

    Array<float> out = createValueArray<float>(odims, 0.f);
    out.eval();

    getQueue().enqueue(kernel::moments<T>, out, in, moment);
    getQueue().sync();
    return out;
}

#define INSTANTIATE(T)                                   \
    template Array<float> moments<T>(const Array<T> &in, \
                                     const af_moment_type moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
