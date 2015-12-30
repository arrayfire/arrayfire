/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <approx.hpp>
#include <kernel/approx1.hpp>
#include <kernel/approx2.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
{

template<typename Ty, typename Tp>
Array<Ty> approx1(const Array<Ty> &in, const Array<Tp> &pos,
                  const af_interp_type method, const float offGrid)
{
    in.eval();
    pos.eval();

    af::dim4 odims = in.dims();
    odims[0] = pos.dims()[0];

    Array<Ty> out = createEmptyArray<Ty>(odims);

    switch(method) {
        case AF_INTERP_NEAREST:
            getQueue().enqueue(kernel::approx1<Ty, Tp, AF_INTERP_NEAREST>,
                               out, in, pos, offGrid);
            break;
        case AF_INTERP_LINEAR:
            getQueue().enqueue(kernel::approx1<Ty, Tp, AF_INTERP_LINEAR>,
                               out, in, pos, offGrid);
            break;
        default:
            break;
    }
    return out;
}


template<typename Ty, typename Tp>
Array<Ty> approx2(const Array<Ty> &in, const Array<Tp> &pos0, const Array<Tp> &pos1,
                  const af_interp_type method, const float offGrid)
{
    in.eval();
    pos0.eval();
    pos1.eval();

    af::dim4 odims = in.dims();
    odims[0] = pos0.dims()[0];
    odims[1] = pos0.dims()[1];

    Array<Ty> out = createEmptyArray<Ty>(odims);

    switch(method) {
        case AF_INTERP_NEAREST:
            getQueue().enqueue(kernel::approx2<Ty, Tp, AF_INTERP_NEAREST>,
                               out, in, pos0, pos1, offGrid);
            break;
        case AF_INTERP_LINEAR:
            getQueue().enqueue(kernel::approx2<Ty, Tp, AF_INTERP_LINEAR>,
                               out, in, pos0, pos1, offGrid);
            break;
        default:
            break;
    }
    return out;
}

#define INSTANTIATE(Ty, Tp)                                                                    \
    template Array<Ty> approx1<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos,              \
                                       const af_interp_type method, const float offGrid);      \
    template Array<Ty> approx2<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos0,             \
                                       const Array<Tp> &pos1, const af_interp_type method,     \
                                       const float offGrid);                                   \

INSTANTIATE(float  , float )
INSTANTIATE(double , double)
INSTANTIATE(cfloat , float )
INSTANTIATE(cdouble, double)

}
