/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/signal.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <approx.hpp>

using af::dim4;
using namespace detail;

namespace {
    template<typename Ty, typename Tp>
    inline void approx1(af_array *yo, const af_array yi,
                        const af_array xo, const int xdim,
                        const Tp &xi_beg, const Tp &xi_step,
                        const af_interp_type method, const float offGrid)
    {
        approx1<Ty>(getArray<Ty>(*yo), getArray<Ty>(yi),
                    getArray<Tp>(xo), xdim,
                    xi_beg, xi_step,
                    method, offGrid);
    }
}

template<typename Ty, typename Tp>
static inline af_array approx2(const af_array zi,
                               const af_array xo, const int xdim, const Tp &xi_beg, const Tp &xi_step,
                               const af_array yo, const int ydim, const Tp &yi_beg, const Tp &yi_step,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(approx2<Ty>(getArray<Ty>(zi),
                                 getArray<Tp>(xo), xdim, xi_beg, xi_step,
                                 getArray<Tp>(yo), ydim, yi_beg, yi_step,
                                 method, offGrid));
}

af_err af_approx1_uniform(af_array *yo, const af_array yi,
                          const af_array xo, const int xdim,
                          const double xi_beg, const double xi_step,
                          const af_interp_type method, const float offGrid)
{
    try {
        const ArrayInfo& yi_info = getInfo(yi);
        const ArrayInfo& xo_info = getInfo(xo);

        const dim4 yi_dims = yi_info.dims();
        const dim4 xo_dims = xo_info.dims();

        ARG_ASSERT(1, yi_info.isFloating());                        // Only floating and complex types
        ARG_ASSERT(2, xo_info.isRealFloating()) ;                   // Only floating types
        ARG_ASSERT(1, yi_info.isSingle() == xo_info.isSingle());    // Must have same precision
        ARG_ASSERT(1, yi_info.isDouble() == xo_info.isDouble());    // Must have same precision
        ARG_ASSERT(3, xdim >= 0 && xdim < 4);

        // POS should either be (x, 1, 1, 1) or (1, yi_dims[1], yi_dims[2], yi_dims[3])
        if (xo_dims[xdim] != xo_dims.elements()) {
            for (int i = 0; i < 4; i++) {
                if (xdim != i) DIM_ASSERT(2, xo_dims[i] == yi_dims[i]);
            }
        }

        ARG_ASSERT(5, xi_step != 0);
        ARG_ASSERT(6, (method == AF_INTERP_CUBIC         ||
                       method == AF_INTERP_CUBIC_SPLINE  ||
                       method == AF_INTERP_LINEAR        ||
                       method == AF_INTERP_LINEAR_COSINE ||
                       method == AF_INTERP_LOWER         ||
                       method == AF_INTERP_NEAREST));

        if (yi_dims.ndims() == 0 || xo_dims.ndims() ==  0) {
            *yo = createHandle(dim4(0,0,0,0), yi_info.getType());
            return AF_SUCCESS;
        }

        dim4 yo_dims = yi_dims;
        yo_dims[xdim] = xo_dims[xdim];
        if (*yo == 0) {
            *yo = createHandle(yo_dims, yi_info.getType());
        }

        DIM_ASSERT(1, getInfo(*yo).dims() == yo_dims);

        switch(yi_info.getType()) {
        case f32: approx1<float  , float >(yo, yi, xo, xdim,
                                           xi_beg, xi_step,
                                           method, offGrid);  break;
        case f64: approx1<double , double>(yo, yi, xo, xdim,
                                           xi_beg, xi_step,
                                           method, offGrid);  break;
        case c32: approx1<cfloat , float >(yo, yi, xo, xdim,
                                           xi_beg, xi_step,
                                           method, offGrid);  break;
        case c64: approx1<cdouble, double>(yo, yi, xo, xdim,
                                                    xi_beg, xi_step,
                                                    method, offGrid);  break;
        default:  TYPE_ERROR(1, yi_info.getType());
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}


af_err af_approx1(af_array *yo, const af_array yi, const af_array xo,
                  const af_interp_type method, const float offGrid)
{
  return af_approx1_uniform(yo, yi, xo, 0, 0.0, 1.0, method, offGrid);
}

af_err af_approx2_uniform(af_array *zo, const af_array zi,
                          const af_array xo, const int xdim, const double xi_beg, const double xi_step,
                          const af_array yo, const int ydim, const double yi_beg, const double yi_step,
                          const af_interp_type method, const float offGrid)
{
    try {
        const ArrayInfo& zi_info = getInfo(zi);
        const ArrayInfo& xo_info = getInfo(xo);
        const ArrayInfo& yo_info = getInfo(yo);

        dim4 zi_dims = zi_info.dims();
        dim4 xo_dims = xo_info.dims();
        dim4 yo_dims = yo_info.dims();

        ARG_ASSERT(1, zi_info.isFloating());                     // Only floating and complex types
        ARG_ASSERT(2, xo_info.isRealFloating());                 // Only floating types
        ARG_ASSERT(4, yo_info.isRealFloating());                 // Only floating types
        ARG_ASSERT(2, xo_info.getType() == yo_info.getType());    // Must have same type
        ARG_ASSERT(1, zi_info.isSingle() == xo_info.isSingle());  // Must have same precision
        ARG_ASSERT(1, zi_info.isDouble() == xo_info.isDouble());  // Must have same precision
        DIM_ASSERT(2, xo_dims == yo_dims);                        // POS0 and POS1 must have same dims

        ARG_ASSERT(3, xdim >= 0 && xdim < 4);
        ARG_ASSERT(5, ydim >= 0 && ydim < 4);
        ARG_ASSERT(7, xi_step != 0);
        ARG_ASSERT(9, yi_step != 0);

        // POS should either be (x, y, 1, 1) or (x, y, zi_dims[2], zi_dims[3])
        if (xo_dims[xdim] * xo_dims[ydim] != xo_dims.elements()) {
            for (int i = 0; i < 4; i++) {
                if (xdim != i && ydim != i) DIM_ASSERT(2, xo_dims[i] == zi_dims[i]);
            }
        }

        if (zi_dims.ndims() == 0 || xo_dims.ndims() ==  0 || yo_dims.ndims() == 0) {
            return af_create_handle(zo, 0, nullptr, zi_info.getType());
        }

        af_array output;

        switch(zi_info.getType()) {
        case f32: output = approx2<float  , float >(zi,
                                                    xo, xdim, xi_beg, xi_step,
                                                    yo, ydim, yi_beg, yi_step,
                                                    method, offGrid);  break;
        case f64: output = approx2<double , double>(zi,
                                                    xo, xdim, xi_beg, xi_step,
                                                    yo, ydim, yi_beg, yi_step,
                                                    method, offGrid);  break;
        case c32: output = approx2<cfloat , float >(zi,
                                                    xo, xdim, xi_beg, xi_step,
                                                    yo, ydim, yi_beg, yi_step,
                                                    method, offGrid);  break;
        case c64: output = approx2<cdouble, double>(zi,
                                                    xo, xdim, xi_beg, xi_step,
                                                    yo, ydim, yi_beg, yi_step,
                                                    method, offGrid);  break;
        default:  TYPE_ERROR(1, zi_info.getType());
        }
        std::swap(*zo, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2(af_array *zo, const af_array zi, const af_array xo, const af_array yo,
                  const af_interp_type method, const float offGrid)
{
  return af_approx2_uniform(zo, zi, xo, 0, 0.0, 1.0, yo, 1, 0.0, 1.0, method, offGrid);
}

