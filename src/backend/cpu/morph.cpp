/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <algorithm>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/morph.hpp>

using af::dim4;

namespace cpu
{
template<typename T, bool isDilation>
Array<T> morph(const Array<T> &in, const Array<T> &mask)
{
    in.eval();
    mask.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::morph<T, isDilation>, out, in, mask);

    return out;
}

template<typename T, bool isDilation>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask)
{
    in.eval();
    mask.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::morph3d<T, isDilation>, out, in, mask);

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> morph  <T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph  <T, false>(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph3d<T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph3d<T, false>(const Array<T> &in, const Array<T> &mask);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )
INSTANTIATE(ushort)
INSTANTIATE(short )
}
