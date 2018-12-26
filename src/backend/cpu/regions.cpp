/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cpu.hpp>
#include <kernel/regions.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <regions.hpp>
#include <af/dim4.hpp>
#include <algorithm>
#include <map>
#include <set>

using af::dim4;

namespace cpu {

template<typename T>
Array<T> regions(const Array<char> &in, af_connectivity connectivity) {
    in.eval();

    Array<T> out = createValueArray(in.dims(), (T)0);
    out.eval();

    getQueue().enqueue(kernel::regions<T>, out, in, connectivity);

    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 af_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
