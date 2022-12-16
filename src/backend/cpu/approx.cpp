/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <approx.hpp>
#include <kernel/approx.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace cpu {

template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const af_interp_type method, const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 1>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 2>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 3>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        default: break;
    }
}

template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const af_interp_type method,
             const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 1>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 2>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 3>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        default: break;
    }
}

#define INSTANTIATE(Ty, Tp)                                       \
    template void approx1<Ty, Tp>(                                \
        Array<Ty> & yo, const Array<Ty> &yi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const af_interp_type method, const float offGrid);        \
    template void approx2<Ty, Tp>(                                \
        Array<Ty> & zo, const Array<Ty> &zi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const Array<Tp> &yo, const int ydim, const Tp &yi_beg,    \
        const Tp &yi_step, const af_interp_type method, const float offGrid);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace cpu
}  // namespace arrayfire
