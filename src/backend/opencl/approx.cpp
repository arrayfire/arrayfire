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

namespace arrayfire {
namespace opencl {
template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const af_interp_type method, const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::approx1<Ty, Tp>(yo, yi, xo, xdim, xi_beg, xi_step, offGrid,
                                    method, 1);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
            kernel::approx1<Ty, Tp>(yo, yi, xo, xdim, xi_beg, xi_step, offGrid,
                                    method, 2);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE:
            kernel::approx1<Ty, Tp>(yo, yi, xo, xdim, xi_beg, xi_step, offGrid,
                                    method, 3);
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
            kernel::approx2<Ty, Tp>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                    yi_beg, yi_step, offGrid, method, 1);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            kernel::approx2<Ty, Tp>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                    yi_beg, yi_step, offGrid, method, 2);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            kernel::approx2<Ty, Tp>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                    yi_beg, yi_step, offGrid, method, 3);
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

}  // namespace opencl
}  // namespace arrayfire
