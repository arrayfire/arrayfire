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
#include <kernel/approx.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename Ty, typename Tp>
    Array<Ty> approx1(const Array<Ty> &yi,
                      const Array<Tp> &xo, const int xdim,
                      const Tp &xi_beg, const Tp &xi_step,
                      const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = yi.dims();
        odims[xdim] = xo.dims()[xdim];

        // Create output placeholder
        Array<Ty> yo = createEmptyArray<Ty>(odims);

        switch(method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::approx1<Ty, Tp, 1> (yo, yi, xo, xdim, xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
            kernel::approx1<Ty, Tp, 2> (yo, yi, xo, xdim, xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE:
            kernel::approx1<Ty, Tp, 3> (yo, yi, xo, xdim, xi_beg, xi_step, offGrid, method);
            break;
        default:
            break;
        }
        return yo;
    }

    template<typename Ty, typename Tp>
    Array<Ty> approx2(const Array<Ty> &zi,
                      const Array<Tp> &xo, const int xdim, const Tp &xi_beg, const Tp &xi_step,
                      const Array<Tp> &yo, const int ydim, const Tp &yi_beg, const Tp &yi_step,
                      const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = zi.dims();
        odims[xdim] = xo.dims()[xdim];
        odims[ydim] = xo.dims()[ydim];

        // Create output placeholder
        Array<Ty> zo = createEmptyArray<Ty>(odims);

        switch(method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::approx2<Ty, Tp, 1> (zo, zi,
                                        xo, xdim, xi_beg, xi_step,
                                        yo, ydim, yi_beg, yi_step,
                                        offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            kernel::approx2<Ty, Tp, 2> (zo, zi,
                                        xo, xdim, xi_beg, xi_step,
                                        yo, ydim, yi_beg, yi_step,
                                        offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            kernel::approx2<Ty, Tp, 3> (zo, zi,
                                        xo, xdim, xi_beg, xi_step,
                                        yo, ydim, yi_beg, yi_step,
                                        offGrid, method);
            break;
        default:
            break;
        }
        return zo;
    }

#define INSTANTIATE(Ty, Tp)                                         \
    template Array<Ty> approx1<Ty, Tp>(const Array<Ty> &yi,         \
                                       const Array<Tp> &xo,         \
                                       const int xdim,              \
                                       const Tp &xi_beg,            \
                                       const Tp &xi_step,           \
                                       const af_interp_type method, \
                                       const float offGrid);        \
    template Array<Ty> approx2<Ty, Tp>(const Array<Ty> &zi,         \
                                       const Array<Tp> &xo,         \
                                       const int xdim,              \
                                       const Tp &xi_beg,            \
                                       const Tp &xi_step,           \
                                       const Array<Tp> &yo,         \
                                       const int ydim,              \
                                       const Tp &yi_beg,            \
                                       const Tp &yi_step,           \
                                       const af_interp_type method, \
                                       const float offGrid);        \

    INSTANTIATE(float  , float )
    INSTANTIATE(double , double)
    INSTANTIATE(cfloat , float )
    INSTANTIATE(cdouble, double)

}
