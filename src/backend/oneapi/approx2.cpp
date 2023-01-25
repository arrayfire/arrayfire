/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <approx.hpp>
#include <err_oneapi.hpp>
#include <kernel/approx2.hpp>

namespace arrayfire {
namespace oneapi {
template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const af_interp_type method,
             const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            if constexpr (!(std::is_same_v<Ty, double> ||
                            std::is_same_v<Ty, cdouble> ||
                            std::is_same_v<Tp, double> ||
                            std::is_same_v<Tp, cdouble>)) {
                kernel::approx2<Ty, Tp, 1>(zo, zi, xo, xdim, xi_beg, xi_step,
                                           yo, ydim, yi_beg, yi_step, offGrid,
                                           method);
            }
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            if constexpr (!(std::is_same_v<Ty, double> ||
                            std::is_same_v<Ty, cdouble> ||
                            std::is_same_v<Tp, double> ||
                            std::is_same_v<Tp, cdouble>)) {
                kernel::approx2<Ty, Tp, 2>(zo, zi, xo, xdim, xi_beg, xi_step,
                                           yo, ydim, yi_beg, yi_step, offGrid,
                                           method);
            }
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            if constexpr (!(std::is_same_v<Ty, double> ||
                            std::is_same_v<Ty, cdouble> ||
                            std::is_same_v<Tp, double> ||
                            std::is_same_v<Tp, cdouble>)) {
                kernel::approx2<Ty, Tp, 3>(zo, zi, xo, xdim, xi_beg, xi_step,
                                           yo, ydim, yi_beg, yi_step, offGrid,
                                           method);
            }
            break;
        default: break;
    }
}

#define INSTANTIATE(Ty, Tp)                                       \
    template void approx2<Ty, Tp>(                                \
        Array<Ty> & zo, const Array<Ty> &zi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const Array<Tp> &yo, const int ydim, const Tp &yi_beg,    \
        const Tp &yi_step, const af_interp_type method, const float offGrid);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace oneapi
}  // namespace arrayfire
