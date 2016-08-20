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
#include <copy.hpp>
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

    const af::dim4 idims = in.dims();
    const af::dim4 mdims = mask.dims();

    const af::dim4 lpad(mdims[0]/2, mdims[1]/2, 0, 0);
    const af::dim4 upad(lpad);
    const af::dim4 odims(lpad[0] + idims[0] + upad[0],
                         lpad[1] + idims[1] + upad[1],
                         idims[2], idims[3]);

    auto out = createEmptyArray<T>(odims);
    auto inp = padArrayBorders(in, lpad, upad, AF_PAD_ZERO);

    getQueue().enqueue(kernel::morph<T, isDilation>, out, inp, mask);

    std::vector<af_seq> idxs(4, af_span);
    idxs[0] = af_seq{double(lpad[0]), double(lpad[0]+idims[0]-1), 1.0};
    idxs[1] = af_seq{double(lpad[1]), double(lpad[1]+idims[1]-1), 1.0};

    return createSubArray(out, idxs);
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
