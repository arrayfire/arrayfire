/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <approx.hpp>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>

#include <af/array.h>
#include <af/defines.h>
#include <af/signal.h>

using af::dim4;
using detail::approx1;
using detail::approx2;
using detail::cdouble;
using detail::cfloat;

namespace {
template<typename Ty, typename Tp>
inline void approx1(af_array *yo, const af_array yi, const af_array xo,
                    const int xdim, const Tp &xi_beg, const Tp &xi_step,
                    const af_interp_type method, const float offGrid) {
    approx1<Ty>(getArray<Ty>(*yo), getArray<Ty>(yi), getArray<Tp>(xo), xdim,
                xi_beg, xi_step, method, offGrid);
}
}  // namespace

template<typename Ty, typename Tp>
inline void approx2(af_array *zo, const af_array zi, const af_array xo,
                    const int xdim, const Tp &xi_beg, const Tp &xi_step,
                    const af_array yo, const int ydim, const Tp &yi_beg,
                    const Tp &yi_step, const af_interp_type method,
                    const float offGrid) {
    approx2<Ty>(getArray<Ty>(*zo), getArray<Ty>(zi), getArray<Tp>(xo), xdim,
                xi_beg, xi_step, getArray<Tp>(yo), ydim, yi_beg, yi_step,
                method, offGrid);
}

void af_approx1_common(af_array *yo, const af_array yi, const af_array xo,
                       const int xdim, const double xi_beg,
                       const double xi_step, const af_interp_type method,
                       const float offGrid, const bool allocate_yo) {
    ARG_ASSERT(0, yo != 0);  // *yo (the af_array) can be null, but not yo
    ARG_ASSERT(1, yi != 0);
    ARG_ASSERT(2, xo != 0);

    const ArrayInfo &yi_info = getInfo(yi);
    const ArrayInfo &xo_info = getInfo(xo);

    const dim4 &yi_dims = yi_info.dims();
    const dim4 &xo_dims = xo_info.dims();
    dim4 yo_dims        = yi_dims;
    yo_dims[xdim]       = xo_dims[xdim];

    ARG_ASSERT(1, yi_info.isFloating());      // Only floating and complex types
    ARG_ASSERT(2, xo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(1, yi_info.isSingle() ==
                      xo_info.isSingle());  // Must have same precision
    ARG_ASSERT(1, yi_info.isDouble() ==
                      xo_info.isDouble());  // Must have same precision
    ARG_ASSERT(3, xdim >= 0 && xdim < 4);

    // POS should either be (x, 1, 1, 1) or (1, yi_dims[1], yi_dims[2],
    // yi_dims[3])
    if (xo_dims[xdim] != xo_dims.elements()) {
        for (int i = 0; i < 4; i++) {
            if (xdim != i) { DIM_ASSERT(2, xo_dims[i] == yi_dims[i]); }
        }
    }

    ARG_ASSERT(5, xi_step != 0);
    ARG_ASSERT(
        6, (method == AF_INTERP_CUBIC || method == AF_INTERP_CUBIC_SPLINE ||
            method == AF_INTERP_LINEAR || method == AF_INTERP_LINEAR_COSINE ||
            method == AF_INTERP_LOWER || method == AF_INTERP_NEAREST));

    if (yi_dims.ndims() == 0 || xo_dims.ndims() == 0) {
        af_create_handle(yo, 0, nullptr, yi_info.getType());
        return;
    }

    if (allocate_yo) { *yo = createHandle(yo_dims, yi_info.getType()); }

    DIM_ASSERT(0, getInfo(*yo).dims() == yo_dims);

    switch (yi_info.getType()) {
        case f32:
            approx1<float, float>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                  offGrid);
            break;
        case f64:
            approx1<double, double>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                    offGrid);
            break;
        case c32:
            approx1<cfloat, float>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                   offGrid);
            break;
        case c64:
            approx1<cdouble, double>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                     offGrid);
            break;
        default: TYPE_ERROR(1, yi_info.getType());
    }
}

af_err af_approx1_uniform(af_array *yo, const af_array yi, const af_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const af_interp_type method,
                          const float offGrid) {
    try {
        af_approx1_common(yo, yi, xo, xdim, xi_beg, xi_step, method, offGrid,
                          true);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx1_uniform_v2(af_array *yo, const af_array yi, const af_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const af_interp_type method,
                             const float offGrid) {
    try {
        ARG_ASSERT(0, yo != 0);  // need to dereference yo in next call
        af_approx1_common(yo, yi, xo, xdim, xi_beg, xi_step, method, offGrid,
                          *yo == 0);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx1(af_array *yo, const af_array yi, const af_array xo,
                  const af_interp_type method, const float offGrid) {
    try {
        af_approx1_common(yo, yi, xo, 0, 0.0, 1.0, method, offGrid, true);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx1_v2(af_array *yo, const af_array yi, const af_array xo,
                     const af_interp_type method, const float offGrid) {
    try {
        ARG_ASSERT(0, yo != 0);  // need to dereference yo in next call
        af_approx1_common(yo, yi, xo, 0, 0.0, 1.0, method, offGrid, *yo == 0);
    }
    CATCHALL;

    return AF_SUCCESS;
}

void af_approx2_common(af_array *zo, const af_array zi, const af_array xo,
                       const int xdim, const double xi_beg,
                       const double xi_step, const af_array yo, const int ydim,
                       const double yi_beg, const double yi_step,
                       const af_interp_type method, const float offGrid,
                       bool allocate_zo) {
    ARG_ASSERT(0, zo != 0);  // *zo (the af_array) can be null, but not zo
    ARG_ASSERT(1, zi != 0);
    ARG_ASSERT(2, xo != 0);
    ARG_ASSERT(6, yo != 0);

    const ArrayInfo &zi_info = getInfo(zi);
    const ArrayInfo &xo_info = getInfo(xo);
    const ArrayInfo &yo_info = getInfo(yo);

    dim4 zi_dims = zi_info.dims();
    dim4 xo_dims = xo_info.dims();
    dim4 yo_dims = yo_info.dims();

    ARG_ASSERT(1, zi_info.isFloating());      // Only floating and complex types
    ARG_ASSERT(2, xo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(4, yo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(2,
               xo_info.getType() == yo_info.getType());  // Must have same type
    ARG_ASSERT(1, zi_info.isSingle() ==
                      xo_info.isSingle());  // Must have same precision
    ARG_ASSERT(1, zi_info.isDouble() ==
                      xo_info.isDouble());  // Must have same precision
    DIM_ASSERT(2, xo_dims == yo_dims);      // POS0 and POS1 must have same dims

    ARG_ASSERT(3, xdim >= 0 && xdim < 4);
    ARG_ASSERT(5, ydim >= 0 && ydim < 4);
    ARG_ASSERT(7, xi_step != 0);
    ARG_ASSERT(9, yi_step != 0);

    // POS should either be (x, y, 1, 1) or (x, y, zi_dims[2], zi_dims[3])
    if (xo_dims[xdim] * xo_dims[ydim] != xo_dims.elements()) {
        for (int i = 0; i < 4; i++) {
            if (xdim != i && ydim != i) {
                DIM_ASSERT(2, xo_dims[i] == zi_dims[i]);
            }
        }
    }

    if (zi_dims.ndims() == 0 || xo_dims.ndims() == 0 || yo_dims.ndims() == 0) {
        af_create_handle(zo, 0, nullptr, zi_info.getType());
        return;
    }

    dim4 zo_dims  = zi_info.dims();
    zo_dims[xdim] = xo_info.dims()[xdim];
    zo_dims[ydim] = xo_info.dims()[ydim];

    if (allocate_zo) { *zo = createHandle(zo_dims, zi_info.getType()); }

    DIM_ASSERT(0, getInfo(*zo).dims() == zo_dims);

    switch (zi_info.getType()) {
        case f32:
            approx2<float, float>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                  yi_beg, yi_step, method, offGrid);
            break;
        case f64:
            approx2<double, double>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                    yi_beg, yi_step, method, offGrid);
            break;
        case c32:
            approx2<cfloat, float>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                   yi_beg, yi_step, method, offGrid);
            break;
        case c64:
            approx2<cdouble, double>(zo, zi, xo, xdim, xi_beg, xi_step, yo,
                                     ydim, yi_beg, yi_step, method, offGrid);
            break;
        default: TYPE_ERROR(1, zi_info.getType());
    }
}

af_err af_approx2_uniform(af_array *zo, const af_array zi, const af_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const af_array yo,
                          const int ydim, const double yi_beg,
                          const double yi_step, const af_interp_type method,
                          const float offGrid) {
    try {
        af_approx2_common(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim, yi_beg,
                          yi_step, method, offGrid, true);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2_uniform_v2(af_array *zo, const af_array zi, const af_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const af_array yo,
                             const int ydim, const double yi_beg,
                             const double yi_step, const af_interp_type method,
                             const float offGrid) {
    try {
        ARG_ASSERT(0, zo != 0);  // need to dereference zo in next call
        af_approx2_common(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim, yi_beg,
                          yi_step, method, offGrid, *zo == 0);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2(af_array *zo, const af_array zi, const af_array xo,
                  const af_array yo, const af_interp_type method,
                  const float offGrid) {
    try {
        af_approx2_common(zo, zi, xo, 0, 0.0, 1.0, yo, 1, 0.0, 1.0, method,
                          offGrid, true);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2_v2(af_array *zo, const af_array zi, const af_array xo,
                     const af_array yo, const af_interp_type method,
                     const float offGrid) {
    try {
        ARG_ASSERT(0, zo != 0);  // need to dereference zo in next call
        af_approx2_common(zo, zi, xo, 0, 0.0, 1.0, yo, 1, 0.0, 1.0, method,
                          offGrid, *zo == 0);
    }
    CATCHALL;

    return AF_SUCCESS;
}
