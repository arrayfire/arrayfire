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
#include <approx.hpp>
#include <kernel/approx.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename Ty, typename Tp>
    Array<Ty> approx1(const Array<Ty> &in, const Array<Tp> &pos,
                      const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos.dims()[0];

        // Create output placeholder
        Array<Ty> out = createEmptyArray<Ty>(odims);
        switch(method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::approx1<Ty, Tp, 1> (out, in, pos, offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
            kernel::approx1<Ty, Tp, 2> (out, in, pos, offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE:
            kernel::approx1<Ty, Tp, 3> (out, in, pos, offGrid, method);
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
        af::dim4 odims = pos0.dims();
        odims[2] = in.dims()[2];
        odims[3] = in.dims()[3];

        // Create output placeholder
        Array<Ty> out = createEmptyArray<Ty>(odims);

        switch(method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::approx2<Ty, Tp, 1> (out, in, pos0, pos1, offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            kernel::approx2<Ty, Tp, 2> (out, in, pos0, pos1, offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            kernel::approx2<Ty, Tp, 3> (out, in, pos0, pos1, offGrid, method);
            break;
        default:
            break;
        }

        return out;
    }

#define INSTANTIATE(Ty, Tp)                                             \
    template Array<Ty> approx1<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos, \
                                       const af_interp_type method, const float offGrid); \
    template Array<Ty> approx2<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos0, \
                                       const Array<Tp> &pos1, const af_interp_type method, \
                                       const float offGrid);            \

    INSTANTIATE(float  , float )
    INSTANTIATE(double , double)
    INSTANTIATE(cfloat , float )
    INSTANTIATE(cdouble, double)

#undef INSTANTIATE
}
